#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include "Utils.h"

namespace batched_gemv {

using cpp_ptx::utils::cp_async_commit_group;
using cpp_ptx::utils::cp_async_gmem_to_smem_zfill;
using cpp_ptx::utils::cp_async_wait_all;
using cpp_ptx::utils::fma_f16x2;
using cpp_ptx::utils::horizontal_add_f16x2;
using cpp_ptx::utils::load_from_scratch;
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
          int WarpsPerBlock = 4,
          int BytesPerLane = 4>
struct GemvKernelTraits
{
    static_assert(std::is_same_v<T, half>,
                  "The PTX GEMV scaffolding currently targets half inputs.");
    static_assert(TileLen > 0 && (TileLen % WarpsPerBlock) == 0,
                  "TileLen must be divisible by the number of warps.");
    static_assert(EmbeddingDim > 0 && (EmbeddingDim % 2) == 0,
                  "EmbeddingDim must be even for packed f16x2 math.");
    static_assert(WarpsPerBlock == 4,
                  "The kernel scaffolding assumes 4 warps per block.");
    static_assert(BytesPerLane == 4,
                  "The current SIMT staging path assumes 4 bytes per lane.");

    using Element = T;

    static constexpr int kWarpSize = 32;
    static constexpr int kWarpsPerBlock = WarpsPerBlock;
    static constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
    static constexpr int kTileLen = TileLen;
    static constexpr int kEmbeddingDim = EmbeddingDim;
    static constexpr int kTileLenPerWarp = kTileLen / kWarpsPerBlock;
    static constexpr int kBytesPerLane = BytesPerLane;
    static constexpr int kElementsPerLaneAccess = kBytesPerLane / sizeof(T);
    static constexpr int kPackedElements = 2;
    static constexpr int kPackedEmbeddingDim = kEmbeddingDim / kPackedElements;
    static constexpr int kQueryLoadsPerLane =
        kEmbeddingDim / (kWarpSize * kElementsPerLaneAccess);
    static constexpr int kRowsPerStagePerWarp = kTileLenPerWarp;
    static constexpr int kPackedRowVectors = kEmbeddingDim / kPackedElements;
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
    int num_work_items = 0;
    int seq_len = 0;
};

template <typename Traits>
__device__ __forceinline__ void stage_query_tile(half const* query_src,
                                                 void* query_smem_base,
                                                 uint32_t lane_idx)
{
    constexpr uint32_t kElementsPerAccess = Traits::kElementsPerLaneAccess;

    for (uint32_t vec_idx = lane_idx; vec_idx < Traits::kEmbeddingDim / kElementsPerAccess;
         vec_idx += Traits::kWarpSize)
    {
        uint32_t const elem_offset = vec_idx * kElementsPerAccess;
        uint32_t const byte_offset = elem_offset * sizeof(half);
        cp_async_gmem_to_smem_zfill<sizeof(uint32_t)>(
            static_cast<char*>(query_smem_base) + byte_offset,
            query_src + elem_offset,
            true);
    }
}

template <typename Traits>
__device__ __forceinline__ void stage_key_tile(half const* key_src,
                                               void* key_smem_base,
                                               uint32_t lane_idx,
                                               uint32_t seq_base,
                                               uint32_t seq_len)
{
    constexpr uint32_t kElementsPerAccess = Traits::kElementsPerLaneAccess;
    constexpr uint32_t kVectorsPerRow =
        Traits::kEmbeddingDim / kElementsPerAccess;
    constexpr uint32_t kTotalVectors =
        Traits::kRowsPerStagePerWarp * kVectorsPerRow;

    for (uint32_t linear_vec = lane_idx; linear_vec < kTotalVectors;
         linear_vec += Traits::kWarpSize)
    {
        uint32_t const row = linear_vec / kVectorsPerRow;
        uint32_t const vec_in_row = linear_vec % kVectorsPerRow;
        uint32_t const global_row = seq_base + row;
        bool const pred = global_row < seq_len;

        uint32_t const elem_offset = vec_in_row * kElementsPerAccess;
        uint32_t const smem_byte_offset =
            (row * Traits::kEmbeddingDim + elem_offset) * sizeof(half);
        uint32_t const gmem_elem_offset =
            global_row * Traits::kEmbeddingDim + elem_offset;

        half const* const gmem_ptr = pred ? key_src + gmem_elem_offset : key_src;
        cp_async_gmem_to_smem_zfill<sizeof(uint32_t)>(
            static_cast<char*>(key_smem_base) + smem_byte_offset, gmem_ptr, pred);
    }
}

template <typename Traits>
__device__ __forceinline__ float compute_dot_product(half const* query_smem_base,
                                                     half const* key_row_smem_base)
{
    half2 accum = make_zero_f16x2();

#pragma unroll
    for (int packed_idx = 0; packed_idx < Traits::kPackedEmbeddingDim;
         ++packed_idx)
    {
        half2 const a =
            reinterpret_cast<half2 const*>(query_smem_base)[packed_idx];
        half2 const b =
            reinterpret_cast<half2 const*>(key_row_smem_base)[packed_idx];
        accum = fma_f16x2(a, b, accum);
    }

    return horizontal_add_f16x2(accum);
}

template <int Stages, typename T, typename Traits = GemvKernelTraits<T>>
__global__ __launch_bounds__(Traits::kThreadsPerBlock) void batched_gemv_kernel(
    T const* query, T const* key, float* output, int seq_len)
{
    using SharedStorage = GemvSharedStorage<T, Stages, Traits>;

    __shared__ SharedStorage shared_storage;

    uint32_t const warp_idx = threadIdx.x / Traits::kWarpSize;
    uint32_t const lane_idx = threadIdx.x & (Traits::kWarpSize - 1);
    uint32_t const work_item_idx = blockIdx.x;

    T const* const query_work_item =
        query + work_item_idx * Traits::kEmbeddingDim;
    T const* const key_work_item =
        key + static_cast<size_t>(work_item_idx) * seq_len * Traits::kEmbeddingDim;
    float* const output_work_item =
        output + static_cast<size_t>(work_item_idx) * seq_len;

    void* const query_stage =
        static_cast<void*>(&shared_storage.query[warp_idx][0]);

    stage_query_tile<Traits>(query_work_item, query_stage, lane_idx);
    cp_async_commit_group();
    cp_async_wait_all();

    for (int seq_tile_base = 0; seq_tile_base < seq_len;
         seq_tile_base += Stages * Traits::kTileLen)
    {
        int const remaining_rows = seq_len - seq_tile_base;
        int const active_stages =
            (remaining_rows + Traits::kTileLen - 1) / Traits::kTileLen;
        int const stages_this_iter = active_stages < Stages ? active_stages : Stages;

        for (int stage_idx = 0; stage_idx < stages_this_iter; ++stage_idx)
        {
            int const stage_seq_base =
                seq_tile_base + stage_idx * Traits::kTileLen +
                static_cast<int>(warp_idx) * Traits::kRowsPerStagePerWarp;
            void* const key_stage =
                static_cast<void*>(&shared_storage.key[stage_idx][warp_idx][0][0]);

            stage_key_tile<Traits>(key_work_item,
                                   key_stage,
                                   lane_idx,
                                   static_cast<uint32_t>(stage_seq_base),
                                   static_cast<uint32_t>(seq_len));
            cp_async_commit_group();
        }

        cp_async_wait_all();

        for (int stage_idx = 0; stage_idx < stages_this_iter; ++stage_idx)
        {
            int const row_idx =
                seq_tile_base + stage_idx * Traits::kTileLen +
                static_cast<int>(warp_idx) * Traits::kRowsPerStagePerWarp +
                static_cast<int>(lane_idx);

            if (row_idx >= seq_len || lane_idx >= Traits::kRowsPerStagePerWarp)
            {
                continue;
            }

            half const* const query_smem_base =
                reinterpret_cast<half const*>(&shared_storage.query[warp_idx][0]);
            half const* const key_row_smem_base =
                reinterpret_cast<half const*>(
                    &shared_storage.key[stage_idx][warp_idx][lane_idx][0]);

            float const result =
                compute_dot_product<Traits>(query_smem_base, key_row_smem_base);
            store_to_global(output_work_item,
                            static_cast<uint32_t>(row_idx),
                            result,
                            true);
        }
    }
}

template <int Stages, typename T, typename Traits = GemvKernelTraits<T>>
inline dim3 get_grid_dim(LaunchParams<T> const& params)
{
    return dim3(static_cast<unsigned>(params.num_work_items), 1u, 1u);
}

template <int Stages, typename T, typename Traits = GemvKernelTraits<T>>
inline dim3 get_block_dim()
{
    return dim3(static_cast<unsigned>(Traits::kThreadsPerBlock), 1u, 1u);
}

template <int Stages, typename T, typename Traits = GemvKernelTraits<T>>
inline cudaError_t launch(LaunchParams<T> const& params,
                          cudaStream_t stream = nullptr)
{
    static_assert(Stages > 0, "Stages must be positive.");

    if (params.query == nullptr || params.key == nullptr || params.output == nullptr)
    {
        return cudaErrorInvalidDevicePointer;
    }
    if (params.num_work_items <= 0 || params.seq_len <= 0)
    {
        return cudaErrorInvalidValue;
    }

    batched_gemv_kernel<Stages, T, Traits>
        <<<get_grid_dim<Stages, T, Traits>(params), get_block_dim<Stages, T, Traits>(), 0, stream>>>(
            params.query, params.key, params.output, params.seq_len);
    return cudaGetLastError();
}

} // namespace batched_gemv
