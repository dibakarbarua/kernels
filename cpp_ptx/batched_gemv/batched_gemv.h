#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "Utils.h"

namespace batched_gemv {

using cpp_ptx::utils::cp_async_commit_group;
using cpp_ptx::utils::cp_async_gmem_to_smem_zfill;
using cpp_ptx::utils::cp_async_wait_all;
using cpp_ptx::utils::fma_f16x2;
using cpp_ptx::utils::half2ToFloat;
using cpp_ptx::utils::make_zero_f16x2;
using cpp_ptx::utils::store_to_global;

/*
----- Batched GEMV Operation to simulate Attention-Decode Kernels ------
Workload:
- We simulate token-by-token decode for LLM(s) using a simulated GEMV
- A = [1,128] B = [seq_len,128], assuming fixed length sequence to avoid workload balancing complexity
- We parallelize across batches and heads, num_work_items = num_batches * num_heads
- Each CTA will work on a full sequence for it's assigned work items
    - We shall assume we have enough work to fully utilize the GPU
- This is a memory bound kernel with an arithmetic intensiy of ~2.
    - We intend to saturate the memory bandwidth of the GPU

Data Movement:
- First we move data from HBM to SMEM and stage A/B tensors for configurable "Stages"
    - This allows us to enforce per-CTA latency hiding
    - Multiple CTAs per SM is another tuning dimension for hiding latency
- A Staging: Multiple sequences per CTA
- B Staging: Multiple tiles per CTA: Latency Hiding

SIMD/SIMT Intrinsics
- We assume each CTA contains 4 warps to fully utilize all Execution Engines
- Each warp works on tile_len/4 individual dot products, no cross-warp dependency
- For one sequence, each warp keeps one register to store sum and then butterfly reduce across lanes
- seq_len is divided into iterations of tile_len tokens each
- NO __syncthreads()! Each warp brings its own data-stage and does not cross compute with another warp.
- A and B are in float16 accuracy and we use FMA f16x2 packed ops per lane
    - To reduce a 128-element hidden dimension each lane has to process 4 elements i.e 2xfp16x2 FMA instructions

Kernel Traits:
- kNumStagesPerWarp is programmed by host
- kTileLen is a kernel trait
- kEmbeddingDim is a kernel trait (reduction-dim)
- kWarpsPerBlock is a kernel trait (4)
- A = 1 x kEmbeddingDim
- B = kTileLen x kEmbeddingDim
- C = 1 x kTileLen
- kBytesPerLane is a kernel trait (4B for NVIDIA SIMT)

Workload Partioning:
- Each Block (CTA) will work on a full sequence
- BatchSize (B) = Number of sequences in workload (num_heads x num_batches in attention terms)
- Sequences per CTA = BatchSize / num_CTAs

Shared Memory:
- A
- B * num_stages

Per-CTA Work:
- The implementation below keeps A warp-private to preserve the no-__syncthreads contract.
- All warps iterate through staged B tiles and compute independently.
- A and B are loaded in 4B chunks per lane and consumed through packed fp16x2 FMA.

PTX Instructions Used:
- cp.async.cg.shared.global
- cp.async.commit_group
- cp.async.wait_group
- fma.rn.f16x2
*/

template <typename T,
          int TileLen = 128,
          int EmbeddingDim = 128,
          int WarpsPerBlock = 4>
struct GemvKernelTraits
{
    static_assert(std::is_same_v<T, half>,
                  "The PTX GEMV scaffolding currently targets half inputs.");
    static_assert(TileLen > 0 && (TileLen % WarpsPerBlock) == 0,
                  "TileLen must be divisible by the number of warps.");
    static_assert(TileLen % 128 == 0, 
                  "With 4 warps/CTA, we need atleast 32 rows per warp in B-matrix to vectorize stores.");
    static_assert(EmbeddingDim > 0 && (EmbeddingDim % 2) == 0,
                  "EmbeddingDim must be even for packed f16x2 math.");
    static_assert(WarpsPerBlock == 4,
                  "The kernel scaffolding assumes 4 warps per block.");

    using Element = T;

    static constexpr int kWarpSize = 32;
    static constexpr int kWarpsPerBlock = WarpsPerBlock;
    static constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
    static constexpr int kTileLen = TileLen;
    static constexpr int kEmbeddingDim = EmbeddingDim;
    static constexpr int kTileLenPerWarp = kTileLen / kWarpsPerBlock;
    static constexpr int kBytesPerLane = sizeof(T) * 2; // packed float/half
    static constexpr int kElemsPerLane = 2; // packed float/half
    static constexpr int kElemsPerWarp = kEmbeddingDim / 2; // packed float/half
    static constexpr int kRowsPerStagePerWarp = kTileLenPerWarp;
};

template <typename T, int Stages, typename Traits = GemvKernelTraits<T>>
struct GemvSharedStorage
{
    static_assert(Stages > 0, "Stages must be positive.");

    alignas(16) T query[Traits::kWarpsPerBlock][Traits::kEmbeddingDim];
    alignas(16)
        T key[Stages][Traits::kWarpsPerBlock][Traits::kRowsPerStagePerWarp]
             [Traits::kEmbeddingDim];
};

template <typename T>
struct LaunchParams
{
    T const* query = nullptr;
    T const* key = nullptr;
    float* output = nullptr;
    int num_batches = 0;
    int seq_len = 0;
};

template <int Stages, typename T, typename Traits = GemvKernelTraits<T>>
__global__ __launch_bounds__(Traits::kThreadsPerBlock) void batched_gemv_kernel(
    T const* query,
    T const* key,
    float* output,
    int num_batches,
    int seq_len)
{
    using SharedStorage = GemvSharedStorage<T, Stages, Traits>;

    __shared__ SharedStorage shared_storage;

    uint32_t const warp_idx = threadIdx.x / Traits::kWarpSize;
    uint32_t const lane_idx = threadIdx.x & (Traits::kWarpSize - 1);

    int const batch_idx_start = static_cast<int>(blockIdx.x);
    int const batch_idx_end = num_batches;
    int const batch_idx_step = static_cast<int>(gridDim.x);

    // OuterLoop: Across batches, CTA-parallel
    for (int batch_idx = batch_idx_start; batch_idx < batch_idx_end; batch_idx += batch_idx_step)
    {
        T const* const query_batch =
            query + static_cast<size_t>(batch_idx) * Traits::kEmbeddingDim;
        T const* const key_batch =
            key + static_cast<size_t>(batch_idx) * seq_len * Traits::kEmbeddingDim;
        float* const output_batch =
            output + static_cast<size_t>(batch_idx) * seq_len;

        void* const query_stage =
            static_cast<void*>(&shared_storage.query[warp_idx][0]);

        // Load inner-dimension of A (vector) one vectorized load at a time
#pragma unroll
        for (uint32_t vec_idx = lane_idx;
             vec_idx < Traits::kElemsPerWarp;
             vec_idx += Traits::kWarpSize)
        {
            uint32_t const elem_offset =
                vec_idx * Traits::kElemsPerLane;
            uint32_t const byte_offset = elem_offset * sizeof(T);
            cp_async_gmem_to_smem_zfill<sizeof(uint32_t)>(
                static_cast<char*>(query_stage) + byte_offset,
                // QUESTION: Why is this at elem offset and not bytes
                query_batch + elem_offset,
                true);
        }
        cp_async_commit_group();
        cp_async_wait_all();

        // InnerLoop: Across sequence, CTA-local
        for (int seq_tile_base = 0; seq_tile_base < seq_len;
             seq_tile_base += Stages * Traits::kTileLen)
        {
            // Single Iteration of B-Matrix Read: tile_rows * stages
#pragma unroll
            for (int stage_idx = 0; stage_idx < Stages; ++stage_idx)
            {
                int const seq_base =
                    seq_tile_base + stage_idx * Traits::kTileLen +
                    static_cast<int>(warp_idx) * Traits::kRowsPerStagePerWarp;
                void* const key_stage =
                    static_cast<void*>(&shared_storage.key[stage_idx][warp_idx][0][0]);

                // Read each row of B-Matrix for current tile-stage
#pragma unroll
                for (uint32_t row_idx = 0; row_idx < Traits::kRowsPerStagePerWarp; row_idx++)
                {
                    uint32_t seq_idx = seq_base + row_idx;
#pragma unroll
                    // Load inner-dimension of B-Matrix-row (vector), one vectorized load at a time
                    for(uint32_t vec_idx = lane_idx;
                        vec_idx < Traits::kElemsPerWarp;
                        vec_idx += Traits::kWarpSize)
                    {
                        bool const pred = seq_idx < seq_len;
                        uint32_t const elem_offset =
                            vec_idx * Traits::kElemsPerLane;
                        uint32_t const smem_byte_offset =
                            (row_idx * Traits::kEmbeddingDim + elem_offset) * sizeof(T);
                        size_t const gmem_elem_offset =
                            static_cast<size_t>(seq_idx) * Traits::kEmbeddingDim +
                            elem_offset;
                        // QUESTION: Why is this at elem offset and not bytes
                        T const* const gmem_ptr =
                            pred ? key_batch + gmem_elem_offset : key_batch;

                        cp_async_gmem_to_smem_zfill<sizeof(uint32_t)>(
                            static_cast<char*>(key_stage) + smem_byte_offset,
                            gmem_ptr,
                            pred);
                    }
                    cp_async_commit_group();
                }
            }
            cp_async_wait_all();

            // Single Iteration of A @ B-Matrix DPR: tile_rows * stages
#pragma unroll
            for (int stage_idx = 0; stage_idx < Stages; ++stage_idx)
            {
                int const seq_base =
                    seq_tile_base + stage_idx * Traits::kTileLen +
                    static_cast<int>(warp_idx) * Traits::kRowsPerStagePerWarp;
                half const* const query_smem_base =
                    reinterpret_cast<half const*>(&shared_storage.query[warp_idx][0]);

                float store_val = 0.0f;
#pragma unroll
                for (uint32_t row_idx = 0; row_idx < Traits::kRowsPerStagePerWarp; row_idx++)
                {
                    half const* const key_row_smem_base =
                        reinterpret_cast<half const*>(
                            &shared_storage.key[stage_idx][warp_idx][row_idx][0]);

                    half2 accum = make_zero_f16x2();

                    // Load and compute FMA for A and B rows, vectorized
#pragma unroll
                    for (int vec_idx = static_cast<int>(lane_idx); vec_idx < Traits::kElemsPerWarp;
                            vec_idx += Traits::kWarpSize)
                    {
                        half2 const a =
                            reinterpret_cast<half2 const*>(query_smem_base)[vec_idx];
                        half2 const b =
                            reinterpret_cast<half2 const*>(key_row_smem_base)[vec_idx];
                        // accum = a * b + accum
                        accum = fma_f16x2(a, b, accum);
                    }

                    float sum = half2ToFloat(accum);

                    // Reduce across warp for final float value
#pragma unroll
                    for (int offset = Traits::kWarpSize / 2; offset > 0; offset >>= 1)
                    {
                        // All lanes have correct sum after loop
                        sum += __shfl_xor_sync(0xffffffffu, sum, offset);
                    }

                    // Predicated register load
                    if (row_idx == lane_idx) {
                        // to store vectorized
                        store_val = sum;
                    }
                    // int const seq_idx = seq_base + static_cast<int>(row_idx);
                    // if (lane_idx == 0 && seq_idx < seq_len)
                    // {
                    //     store_to_global(output_batch,
                    //                     static_cast<uint32_t>(seq_idx),
                    //                     sum,
                    //                     true);
                    // }
                }
                store_to_global(output_batch + seq_base, lane_idx, store_val, (seq_base + lane_idx) < seq_len);
            }
        }
    }
}

} // namespace batched_gemv
