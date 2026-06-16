#pragma once

#include <cstddef>
#include <cuda_runtime.h>

__global__ __launch_bounds__(256) void tiled_dequant_float4_kernel(
        const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE)
{
    constexpr int kElementsPerVector = 4;

    int thread_idx = static_cast<int>(threadIdx.x);
    int block_idx_x = static_cast<int>(blockIdx.x);
    int block_idx_y = static_cast<int>(blockIdx.y);
    int block_dim_x = static_cast<int>(blockDim.x);
    int grid_dim_x = static_cast<int>(gridDim.x);
    int vectors_per_tile_row = TILE_SIZE / kElementsPerVector;
    int rows_per_block = block_dim_x / vectors_per_tile_row;
    int local_row = thread_idx / vectors_per_tile_row;
    int lane_in_row = thread_idx % vectors_per_tile_row;
    int row_idx = block_idx_y * rows_per_block + local_row;

    if (row_idx >= M) {
        return;
    }

    int scale_idx = (row_idx / TILE_SIZE) * grid_dim_x + block_idx_x;
    float r_scale = S[scale_idx];
    int tile_col = block_idx_x * TILE_SIZE;
    int remaining_cols = N - tile_col;
    int tile_width =
            remaining_cols < TILE_SIZE ? remaining_cols : TILE_SIZE;
    int row_start_idx = row_idx * N + tile_col;

    // CUDA allocations are at least 16-byte aligned. Advance to the next
    // four-float boundary so both the vector load and store remain aligned.
    int alignment_prefix =
            (kElementsPerVector - (row_start_idx % kElementsPerVector)) %
            kElementsPerVector;
    int scalar_prefix =
            alignment_prefix < tile_width ? alignment_prefix : tile_width;
    int vector_count =
            (tile_width - scalar_prefix) / kElementsPerVector;
    int scalar_tail_start =
            scalar_prefix + vector_count * kElementsPerVector;
    int scalar_tail = tile_width - scalar_tail_start;

    if (lane_in_row < scalar_prefix) {
        int input_idx = row_start_idx + lane_in_row;
        Y[input_idx] = X[input_idx] * r_scale;
    }

    if (lane_in_row < vector_count) {
        int input_idx =
                row_start_idx + scalar_prefix +
                lane_in_row * kElementsPerVector;
        float4 r_input =
                *reinterpret_cast<const float4*>(X + input_idx);
        float4 r_output = {
                r_input.x * r_scale,
                r_input.y * r_scale,
                r_input.z * r_scale,
                r_input.w * r_scale,
        };
        *reinterpret_cast<float4*>(Y + input_idx) = r_output;
    }

    if (lane_in_row < scalar_tail) {
        int input_idx =
                row_start_idx + scalar_tail_start + lane_in_row;
        Y[input_idx] = X[input_idx] * r_scale;
    }
}

// X, S, Y are device pointers
extern "C" void solve(const float* X, const float* S, float* Y, int M, int N, int TILE_SIZE) {
    constexpr int kThreadsPerBlock = 256;
    constexpr int kElementsPerVector = 4;

    if (M <= 0 || N <= 0 || TILE_SIZE <= 0 ||
        TILE_SIZE % kElementsPerVector != 0 ||
        (kThreadsPerBlock * kElementsPerVector) % TILE_SIZE != 0) {
        return;
    }

    int rows_per_block =
            kThreadsPerBlock * kElementsPerVector / TILE_SIZE;
    int num_blocks_x = (N + TILE_SIZE - 1) / TILE_SIZE;
    int num_blocks_y = (M + rows_per_block - 1) / rows_per_block;
    dim3 grid_dim(static_cast<unsigned int>(num_blocks_x),
                  static_cast<unsigned int>(num_blocks_y));

    tiled_dequant_float4_kernel<<<grid_dim, kThreadsPerBlock>>>(
            X, S, Y, M, N, TILE_SIZE);
}
