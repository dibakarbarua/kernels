#pragma once

#include <cuda.h>

#include <cassert>
#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>

#include "Utils.h"

/*
---- Swizzling ----
- SRAM modules in ML ASICs are usually banked at a high width (4B-16B or more) to allow high throughput access per thread/lane
- These banks are of a fixed number (for example 32x4B banks in NVIDIA GPUs) after which the banks repeat
- For accessing data from a single warp/SIMD operation this leads to contiguous addresses and no conflicts if all lanes access 32B
- But there are two cases when ther e can be bank-conflicts:
    - Transposed Access (SIMD operations read/write strided i.e a full column)
    - Concurrent Access of different rows by different SIMD Units or FF Units (TensorCores in NVIDIA)

There are two primary methods of solving this issue:
    - padding by one bank in x-dim to force rotation on strides
    - Swizzling addresses as they cross stride boundaries

Swizzling an SMEM address space (by NVIDIA)
NVIDIA published an (awesome) Swizzle Functor with their Cutlass 3.0 release.
- It basically has three aspects: MBase, SShift and BBits.
- The idea is to inject higher order bits into lower order bits so that strided addresses 
... (which have common lower order bits) do not collide when mapped to a small address space (such as 32x32B banks)

address = | ---------------------------------- <--- B-bits ----> ------------|
   mask = | -------- <--- B-bits ----> ------- <--- B-bits ----> ------------|
          | -------------------------< ----- SShift ---------- ><---MBase--->|

The B-Bits in the original address (starting at MBase) will be XOR'ed with the shifted B-bits from higher-bits (starting at SShift + MBase)
- Hence addresses will retain their original addressing for 2^(M+S) Bytes
    - This is the size of one SMEM row, as we can keep this unchanged as the banks rotate by nature every 4B
- After 2^(M+S) bytes, we will toggle the bits starting at MBase.
- MBase helps use define how many Bytes we wish to be contiguous and hence not rotated in-between
    - This will be dictated by how many bytes are access per SIMD/SIMT lane that should be contiguous
    - The maximum size of a Load per Lane in NVIDIA GPUs is 16B, hence MBase = 4.
- For 16B per lane and 32 lanes we get SMEM row = 512B, hence 2^(M+S) = 9. Hence S = 5.
- For  4B per lane and 32 lanes we get SMEM row = 128B, hence 2^(M+S) = 7. Hence S = 3.
Hence, 2^(M+S) covers one SMEM row and for 2^B rows we keep swizzling addresses.
Hence number of rows = 2^B
For a 32x32 transpose of 4B elements, 2^B = 32 and B = 5.

For our case, we can plan for 16B access per lane and multiples of 32 rows per SMEM allocation.
Hence,
MBase = 4
SShift = 3
BBits = {5 for 32 rows; 6 for 64 rows}
*/

/*
------ Per-Lane Layouts ------
If we do a 32x32 transpose of 4B elements, we can write transposed into SMEM and read column-wise and write back to GMEM all coalesced.
Assuming we want to load at atleast 4B per lane to saturate one cacheline (usually 128B), 
    we will need to do per-lane transpose for smaller elements than 4B.

--- 32x32x4B ---
[ e0,0(4B) e0,1(4B) ....... e0,31(4B)]
[ e1,0(4B) e1,1(4B) ....... e1,31(4B)]
...
We can see that on column-read one SIMD/SIMT operation will read contiguous destination elements.

--- 64x64x2B ---
[ e0,0|e0,1(4B) e0,2|e0,3(4B) ....... e0,30|e0,31(4B) ]
[ e1,0|e1,1(4B) e1,2|e1,3(4B) ....... e1,30|e1,31(4B) ]
....
If we read this column-wise 4B per lane, we get e0,0 e0,1 e1,0 e1,1 which is incorrect
Here each lane will have to do a 2x2 transpose within it's 2 registers holding e0,0 e0,1 (4B) and e1,0 e1,1 (4B)

--- 128x128x1B ---
[ e0,0|e0,1|e0,2|e0,3(4B)  ....... e0,28|e0,29|e0,30|e0,31(4B) ]
[ e1,0|e1,1|e1,2|e1,3(4B) ........ e1,28|e1,29|e1,30|e1,31(4B) ]
[ e2,0|e2,1|e2,2|e2,3(4B) ........ e2,28|e2,29|e2,30|e2,31(4B) ]
[ e3,0|e3,1|e3,2|e3,3(4B) ........ e3,28|e3,29|e3,30|e3,31(4B) ]
....
Here each lane will have to do a 4x4 transpose within it's 4 registers
*/

/*
------ Hiding Global Memory Latency ------
In any SIMD engine, we can usually have a number of concurrent operations using the ALUs (replicated VALUs)
In NVIDIA's SIMT engine, this is 4xwarps per SM.
Each $-line requested by an operation (warp) incurs a GMEM latency.
    Since this is a streaming kernel (like most memory bound kernels that do not reuse data),
    that latency will be from HBM (usually around 1000 cycles)
Hence total_cachelines_reqd * 128B = SIMT_Engine_BW_in_BytesPerCycle * 1000Cycles
Let's define that as a tunable macro = BW
Hence, total_cachelines_reqd = BW * 1024/128 = BW * 8

We find that a full transpose (in elements) for one warp/SIMD engine is:
    - 32x32 for 4B elements
    - 64x64 for 2B elements
    - 128x128 for 1B elements
Total required cacheline reads is:
    - 32 for 4B elements
    - 64 for 2B elements
    - 128 for 1B elements
Hence for 4 warps/SIMD instructions, total required cacheline reads is:
    - 128 (4B)
    - 256 (2B)
    - 512 (1B)

For more inflight requests, we can always double or N-buffer in SMEM.
Here are some common per SM HBM bandwidth metrics from NVIDIA GPUs:
Arch	    Bytes / cycle   CacheLines Outstanding
A100	    ~13             104
H100	    ~14             112
H200	    ~20             160
Blackwell	~20–22          180

Hence with 4 concurrent warps/SIMD ops we can largely hide the GMEM latency for even 4B transpose in A100/H100
But in H200/Blackwell we will need to double buffer for 4B elements.
For 2B/1B elements, 256/512 total cachelines outstanding per SM is sufficient!
*/

/*
------- Hiding Shared Memory Latency --------
Nominally the SMEM datapath incurs a latency of ~32 cycles per Load.
With a minimum of 32 cachelines outstanding per SM, that latency is also hidden in our case.
*/

namespace swizzled_transpose {

using cpp_ptx::utils::cp_async_commit_group;
using cpp_ptx::utils::cp_async_gmem_to_smem_zfill;
using cpp_ptx::utils::cp_async_wait_group;
using cpp_ptx::utils::load_from_scratch;
using cpp_ptx::utils::store_to_global;
using cpp_ptx::utils::swizzle_smem_offset;

template <typename T>
struct TransposeTileTraits
{
    static_assert(std::is_trivially_copyable_v<T>,
                  "TransposeTileTraits requires a trivially copyable type.");
    static_assert(sizeof(T) == 1 || sizeof(T) == 2 || sizeof(T) == 4,
                  "Only 1B, 2B, and 4B element types are currently supported.");

    static constexpr int kWarpSize = 32;
    static constexpr int kWarpsPerBlock = 4;
    static constexpr int kThreadsPerBlock = kWarpSize * kWarpsPerBlock;
    static constexpr int kBytesPerLaneAccess = 4;
    static constexpr int kElementsPerLaneAccess = kBytesPerLaneAccess / sizeof(T);
    static constexpr int kTileExtent = kWarpSize * kElementsPerLaneAccess;
    static constexpr int kTileRows = kTileExtent;
    static constexpr int kTileCols = kTileExtent;
    static constexpr int kElementsPerWarp = kTileRows * kTileCols;
};

template <typename T, uint32_t SwizzleB, uint32_t SwizzleM, uint32_t SwizzleS>
__device__ __forceinline__ void load_stage_to_warp_smem(
    T const* input,
    void* smem_stage_base,
    uint32_t tile_start_x,
    uint32_t tile_start_y,
    uint32_t lane_idx,
    uint32_t warp_idx,
    uint32_t stage_idx,
    uint32_t rows,
    uint32_t cols,
    uint32_t rows_per_tile,
    uint32_t cols_per_tile,
    uint32_t rows_per_warp,
    uint32_t rows_per_stage)
{
    for (uint32_t row_in_stage = 0; row_in_stage < rows_per_warp; ++row_in_stage)
    {
        uint32_t const stage_row_base = stage_idx * rows_per_stage;
        uint32_t const warp_row_base = warp_idx * rows_per_warp;
        uint32_t const warp_row = stage_row_base + warp_row_base + row_in_stage;
        uint32_t const gmem_load_y = tile_start_y + warp_row;
        uint32_t const gmem_load_x = tile_start_x + lane_idx;
        bool const gmem_load_pred = (gmem_load_y < rows) && (gmem_load_x < cols);

        uint32_t const gmem_load_offset = gmem_load_y * cols + gmem_load_x;

        // Each warp transposes its own 32x32 tile in a private SMEM region.
        uint32_t const smem_write_y = lane_idx;
        uint32_t const smem_write_x = row_in_stage;
        uint32_t const smem_write_linear_offset =
            (smem_write_y * cols_per_tile + smem_write_x) * sizeof(T);
        uint32_t const smem_write_offset =
            swizzle_smem_offset<SwizzleB, SwizzleM, SwizzleS>(
                smem_write_linear_offset);

        T const* const gmem_load_ptr = gmem_load_pred ? input + gmem_load_offset : input;
        void* const smem_write_ptr =
            reinterpret_cast<char*>(smem_stage_base) + smem_write_offset;

        cp_async_gmem_to_smem_zfill<sizeof(T)>(smem_write_ptr,
                                               static_cast<void const*>(gmem_load_ptr),
                                               gmem_load_pred);
    }
}

template <typename T, uint32_t SwizzleB, uint32_t SwizzleM, uint32_t SwizzleS>
__device__ __forceinline__ void store_transposed_stage_from_warp_smem(
    void const* smem_stage_base,
    T* output,
    uint32_t tile_start_x,
    uint32_t tile_start_y,
    uint32_t lane_idx,
    uint32_t warp_idx,
    uint32_t stage_idx,
    uint32_t rows,
    uint32_t cols,
    uint32_t rows_per_tile,
    uint32_t cols_per_tile,
    uint32_t rows_per_warp,
    uint32_t rows_per_stage)
{
    (void)rows_per_tile;

    for (uint32_t row_in_stage = 0; row_in_stage < rows_per_warp; ++row_in_stage)
    {
        uint32_t const stage_row_base = stage_idx * rows_per_stage;
        uint32_t const warp_row_base = warp_idx * rows_per_warp;
        uint32_t const gmem_store_y = tile_start_x + row_in_stage;
        // NOTE: stage_row_base + warp_row_base is an important offset to get right
        // We need to be at the correct y-tile and then we index by lane
        uint32_t const gmem_store_x = tile_start_y + stage_row_base + warp_row_base +
                                      lane_idx;
        bool const gmem_store_pred = (gmem_store_y < cols) && (gmem_store_x < rows);

        uint32_t const gmem_store_offset = gmem_store_y * rows + gmem_store_x;

        uint32_t const smem_read_y = row_in_stage;
        uint32_t const smem_read_x = lane_idx;
        uint32_t const smem_read_linear_offset =
            (smem_read_y * cols_per_tile + smem_read_x) * sizeof(T);
        uint32_t const smem_read_offset =
            swizzle_smem_offset<SwizzleB, SwizzleM, SwizzleS>(
                smem_read_linear_offset);

        T const value = load_from_scratch<T>(smem_stage_base, smem_read_offset);
        store_to_global(output, gmem_store_offset, value, gmem_store_pred);
    }
}

template <typename T, int StagesPerWarp>
struct TransposeSharedStorage
{
    using Element = T;
    using Traits = TransposeTileTraits<T>;

    static constexpr int kStagesPerWarp = StagesPerWarp;
    static constexpr int kWarpsPerBlock = Traits::kWarpsPerBlock;
    static constexpr int kTileRows = Traits::kTileRows;
    static constexpr int kTileCols = Traits::kTileCols;
    static constexpr int kRowsPerWarp = kTileRows;
    static constexpr int kElementsPerWarpStage = kRowsPerWarp * kTileCols;

    alignas(16) Element
        smem[kStagesPerWarp][kWarpsPerBlock][kRowsPerWarp][kTileCols];
};

template <typename T, int StagesPerWarp>
__global__ __launch_bounds__(TransposeTileTraits<T>::kThreadsPerBlock)
void transpose_kernel(T const* input, T* output, int rows, int cols)
{
    using Traits = TransposeTileTraits<T>;
    using SharedStorage = TransposeSharedStorage<T, StagesPerWarp>;

    static_assert(Traits::kWarpsPerBlock == 4,
                  "This transpose kernel currently supports exactly 4 warps.");
    static_assert(sizeof(T) == 4,
                  "transpose_kernel currently supports only 4-byte element types.");
    static_assert(Traits::kTileRows == 32 && Traits::kTileCols == 32,
                  "The 4-byte transpose kernel expects a 32x32 tile.");

    assert((blockDim.x * blockDim.y * blockDim.z) == Traits::kThreadsPerBlock);

    __shared__ SharedStorage shared_storage;

    constexpr uint32_t kSwizzleB = 5;
    constexpr uint32_t kSwizzleM = 4;
    constexpr uint32_t kSwizzleS = 3;
    constexpr uint32_t kRowsPerTile = Traits::kTileRows;
    constexpr uint32_t kColsPerTile = Traits::kTileCols;
    constexpr uint32_t kRowsPerWarp = kRowsPerTile;
    // Each stage covers one 32x32 tile per warp, stacked along the input-row axis.
    constexpr uint32_t kRowsPerStage = kRowsPerWarp * Traits::kWarpsPerBlock;
    constexpr uint32_t kRowsPerCtaIteration = kRowsPerStage * StagesPerWarp;

    uint32_t const cta_tile_idx_x = blockIdx.x;
    uint32_t const cta_tile_idx_y = blockIdx.y;
    uint32_t const cta_tile_start_x = cta_tile_idx_x * kColsPerTile;
    uint32_t const cta_tile_start_y = cta_tile_idx_y * kRowsPerCtaIteration;
    uint32_t const cta_tile_step_x = gridDim.x * kColsPerTile;
    uint32_t const cta_tile_step_y = gridDim.y * kRowsPerCtaIteration;
    uint32_t const lane_idx = threadIdx.x & (Traits::kWarpSize - 1);
    uint32_t const warp_idx = threadIdx.x / Traits::kWarpSize;

    uint32_t const rows_u32 = static_cast<uint32_t>(rows);
    uint32_t const cols_u32 = static_cast<uint32_t>(cols);

    for (uint32_t tile_start_y = cta_tile_start_y; tile_start_y < rows_u32;
         tile_start_y += cta_tile_step_y)
    {
        for (uint32_t tile_start_x = cta_tile_start_x; tile_start_x < cols_u32;
             tile_start_x += cta_tile_step_x)
        {
            for (uint32_t stage_idx = 0; stage_idx < StagesPerWarp; ++stage_idx)
            {
                void* const smem_stage_base =
                    static_cast<void*>(&shared_storage.smem[stage_idx][warp_idx][0][0]);

                load_stage_to_warp_smem<T, kSwizzleB, kSwizzleM, kSwizzleS>(
                    input,
                    smem_stage_base,
                    tile_start_x,
                    tile_start_y,
                    lane_idx,
                    warp_idx,
                    stage_idx,
                    rows_u32,
                    cols_u32,
                    kRowsPerTile,
                    kColsPerTile,
                    kRowsPerWarp,
                    kRowsPerStage);

                cp_async_commit_group();
            }

            // IMPORTANT!: This is sub-optimal, we only need to wait for one-stage before transpose
            cp_async_wait_group<1>();

            for (uint32_t stage_idx = 0; stage_idx < StagesPerWarp; ++stage_idx)
            {
                void const* const smem_stage_base_const =
                    static_cast<void const*>(
                        &shared_storage.smem[stage_idx][warp_idx][0][0]);

                store_transposed_stage_from_warp_smem<
                    T, kSwizzleB, kSwizzleM, kSwizzleS>(
                    smem_stage_base_const,
                    output,
                    tile_start_x,
                    tile_start_y,
                    lane_idx,
                    warp_idx,
                    stage_idx,
                    rows_u32,
                    cols_u32,
                    kRowsPerTile,
                    kColsPerTile,
                    kRowsPerWarp,
                    kRowsPerStage);
            }
        }
    }
}

} // namespace swizzled_transpose
