#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

template <std::size_t ITERS_PER_BLOCK, std::size_t NUM_WARPS_PER_BLOCK>
__global__ __launch_bounds__(1024) void histogram_kernel(const int* __restrict__ input,
                                                          int* __restrict__ histogram,
                                                          int N,
                                                          int num_bins) {
    auto block_idx = blockIdx.x;
    auto NUM_BLOCKS = gridDim.x;
    constexpr int NUM_THREADS_PER_WARP = 32;
    constexpr int ELEMS_PER_PACKED_LOAD = 2;
    constexpr int ELEMS_PER_WARP = ELEMS_PER_PACKED_LOAD * NUM_THREADS_PER_WARP;
    constexpr int ELEMS_PER_BLOCK = ELEMS_PER_WARP * NUM_WARPS_PER_BLOCK;
    auto warp_idx = threadIdx.x / NUM_THREADS_PER_WARP;
    auto thread_idx = threadIdx.x % NUM_THREADS_PER_WARP;

    // each CTA works on discrete input elements
    // we spawn as many CTAs as there are tile blocks in input
    // NUM_BLOCKS = ceil(N / (ELEMS_PER_BLOCK * ITERS_PER_BLOCK))
    uint32_t input_start_idx = 
        ELEMS_PER_BLOCK * block_idx
        + ELEMS_PER_WARP * warp_idx
        + ELEMS_PER_PACKED_LOAD * thread_idx;
    
    uint32_t input_step_idx = ELEMS_PER_BLOCK * NUM_BLOCKS;

    // warp-private histogram for every warp in CTA
    // size = histogram[NUM_WARPS_PER_BLOCK][num_bins]
    extern __shared__ int smem_pool[];
    int* histogram_private = smem_pool;
    int* histogram_shared = &smem_pool[NUM_WARPS_PER_BLOCK * num_bins];

    // zero-out all private histogram entries
    int* hist_ptr = histogram_private + warp_idx * num_bins;
    for (uint32_t hist_idx = thread_idx; hist_idx < num_bins; hist_idx += NUM_THREADS_PER_WARP) {
        hist_ptr[hist_idx] = 0;
    }
    __syncwarp();

    // zero-out all shared histogram entries
    int* hist_shared_ptr = histogram_shared;
    if (warp_idx == 0) {
        for (uint32_t hist_idx = thread_idx; hist_idx < num_bins; hist_idx += NUM_THREADS_PER_WARP) {
            hist_shared_ptr[hist_idx] = 0;
        }
    }

    uint64_t reg_input[ITERS_PER_BLOCK];
    uint64_t reg_output_bin[ITERS_PER_BLOCK];
    uint32_t valid_count[ITERS_PER_BLOCK];
    uint32_t const input_size = static_cast<uint32_t>(N);
    uint32_t const bin_count = static_cast<uint32_t>(num_bins);
    uint32_t const bin_mask32 = bin_count - 1;
    uint64_t const bin_mask =
        static_cast<uint64_t>(bin_mask32) |
        (static_cast<uint64_t>(bin_mask32) << 32);
    bool const num_bins_is_power_of_two = (bin_count & (bin_count - 1)) == 0;

#pragma unroll
    for (uint32_t iter = 0; iter < ITERS_PER_BLOCK; iter++) {
        uint32_t const input_idx = input_start_idx + iter * input_step_idx;
        if (input_idx + 1 < input_size) {
            reg_input[iter] =
                *reinterpret_cast<uint64_t const*>(input + input_idx);
            valid_count[iter] = 2;
        } else if (input_idx < input_size) {
            reg_input[iter] =
                static_cast<uint64_t>(static_cast<uint32_t>(input[input_idx]));
            valid_count[iter] = 1;
        } else {
            reg_input[iter] = 0;
            valid_count[iter] = 0;
        }
    }

    if (num_bins_is_power_of_two) {
#pragma unroll
        for (uint32_t iter = 0; iter < ITERS_PER_BLOCK; iter++) {
            reg_output_bin[iter] = reg_input[iter] & bin_mask;
        }
    } else {
#pragma unroll
        for (uint32_t iter = 0; iter < ITERS_PER_BLOCK; iter++) {
            uint32_t const bin_low =
                static_cast<uint32_t>(reg_input[iter]) % bin_count;
            uint32_t const bin_high =
                static_cast<uint32_t>(reg_input[iter] >> 32) % bin_count;
            reg_output_bin[iter] =
                static_cast<uint64_t>(bin_low) |
                (static_cast<uint64_t>(bin_high) << 32);
        }
    }

    int* warp_histogram = histogram_private + warp_idx * num_bins;
#pragma unroll
    for (uint32_t iter = 0; iter < ITERS_PER_BLOCK; iter++) {
        uint32_t const bin_low = static_cast<uint32_t>(reg_output_bin[iter]);
        uint32_t const bin_high = static_cast<uint32_t>(reg_output_bin[iter] >> 32);
        if (valid_count[iter] > 0) {
            atomicAdd(warp_histogram + bin_low, 1);
        }
        if (valid_count[iter] > 1) {
            atomicAdd(warp_histogram + bin_high, 1);
        }
    }

    // once all warps are done with all elements, we reduce across warps in CTA
    __syncthreads();
    
    // reduce warp-private histograms into a CTA-private histogram
    hist_ptr = histogram_private + warp_idx * num_bins;
    hist_shared_ptr = histogram_shared;
    for (uint32_t hist_idx = thread_idx; hist_idx < num_bins; hist_idx += NUM_THREADS_PER_WARP) {
        atomicAdd(hist_shared_ptr + hist_idx, hist_ptr[hist_idx]);
    }

    // CTA reduced histogram ready
    __syncthreads();

    // write out final histogram in GMEM
    // combine CTA-private histograms in global memory
    hist_shared_ptr = histogram_shared;
    for (uint32_t hist_idx = threadIdx.x; hist_idx < num_bins; hist_idx += blockDim.x) {
        atomicAdd(histogram + hist_idx, hist_shared_ptr[hist_idx]);
    }
}

// input, histogram are device pointers
extern "C" void solve(const int* input, int* histogram, int N, int num_bins) {
    if (num_bins <= 0) {
        return;
    }

    cudaMemset(histogram, 0, static_cast<std::size_t>(num_bins) * sizeof(int));

    if (N <= 0) {
        return;
    }

    constexpr int NUM_WARPS_PER_BLOCK = 4;
    constexpr int NUM_THREADS_PER_WARP = 32;
    constexpr int ELEMS_PER_PACKED_LOAD = 2;
    constexpr int block_size = NUM_WARPS_PER_BLOCK * NUM_THREADS_PER_WARP;
    constexpr int ITERS_PER_THREAD = 4;
    constexpr int tile_size = block_size * ELEMS_PER_PACKED_LOAD * ITERS_PER_THREAD;
    int grid_size = (N + tile_size - 1) / tile_size;

    std::size_t smem_size =
        static_cast<std::size_t>(NUM_WARPS_PER_BLOCK + 1) *
        static_cast<std::size_t>(num_bins) * sizeof(int);
    histogram_kernel<ITERS_PER_THREAD, NUM_WARPS_PER_BLOCK>
        <<<grid_size, block_size, smem_size>>>(input, histogram, N, num_bins);
}
