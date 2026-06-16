#pragma once

#include <cstddef>
#include <cuda_runtime.h>

template <std::size_t ROWS_PER_WARP>
__global__ __launch_bounds__(1024) void kernel_tsz128(
        const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE)
{
    // registers
    float2 r_scale;
    float2 r_input[ROWS_PER_WARP];
    float2 r_output[ROWS_PER_WARP];

    // shared
    __shared__ float s_scale;

    if (threadIdx.x == 0) {
        // TODO: Use cutlass fast divmod
        auto block_idx_x = blockIdx.x % TILE_SIZE;
        auto block_idx_y = blockIdx.x / TILE_SIZE;
        s_scale = *(S + block_idx_y * TILE_SIZE + block_idx_x);
    }
    __syncthreads();

    r_scale = {s_scale, s_scale}; // broadcast

    // thread-value indexing
    // TODO: Use cutlass fast divmod
    auto block_idx_y = blockIdx.x / TILE_SIZE;
    auto block_idx_x = blockIdx.x % TILE_SIZE;
    auto warp_idx_y = threadIdx.x / 64; // 2 warps per row
    auto thread_idx_x = threadIdx.x % 64; // 2 warps per row
    int row_idx = block_idx_y * TILE_SIZE + warp_idx_y * ROWS_PER_WARP;
    int col_idx = block_idx_x * TILE_SIZE + thread_idx_x * 2; // packed float2 elements

#pragma unroll
    for (std::size_t row = 0; row < ROWS_PER_WARP; ++row) {
        r_input[row] =
                *reinterpret_cast<const float2*>(X + (row_idx + row) * N + col_idx);
    }
#pragma unroll
    for (std::size_t row = 0; row < ROWS_PER_WARP; ++row) {
        r_output[row].x = r_input[row].x * r_scale.x;
        r_output[row].y = r_input[row].y * r_scale.y;
    }
#pragma unroll
    for (std::size_t row = 0; row < ROWS_PER_WARP; ++row) {
        *reinterpret_cast<float2*>(Y + (row_idx + row) * N + col_idx) = r_output[row];
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    // each block uses same scale tile [TILE_SIZE][TILE_SIZE]
    int tile_area = TILE_SIZE * TILE_SIZE;
    int num_blocks = (M * N + tile_area - 1) / tile_area;
    int num_warps_x = 2; // for TILE_SIZE = 128, 2xelements per thread, 32 threads
    int rows_per_warp = 8; // tunable

    if (TILE_SIZE == 16) {
        num_warps_x = 1; // halfwarp
        rows_per_warp *= 4; // halfwarp per row
        return; // not implemented
    }
    else if (TILE_SIZE == 32) {
        num_warps_x = 1; // fullwarp
        rows_per_warp *= 2; // fullwarp per row
        return; // not implemented
    }
    else if (TILE_SIZE == 64) {
        num_warps_x = 1; // fullwarp with packed float ops
        rows_per_warp *= 2; // fullwarp
        return; // not implemented
    }

    int num_warps_y = TILE_SIZE/rows_per_warp;
    int block_size = num_warps_x * num_warps_y * 32;
    if (rows_per_warp == 8) {
        kernel_tsz128<8><<<num_blocks, block_size>>>(X, S, Y, M, N, 128);
    } else {
        return; // not implemented
    }
    return;
}
