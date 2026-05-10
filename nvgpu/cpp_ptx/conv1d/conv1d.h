#include <cstdint>
#include <cuda_runtime.h>

/*
----- Question -----
Implement a program that performs a 1D convolution operation. Given an input array and a kernel (filter), compute the convolved output. The convolution should be performed with a "valid" boundary condition, meaning the kernel is only applied where it fully overlaps with the input.

The input consists of two arrays:

input: A 1D array of 32-bit floating-point numbers.
kernel: A 1D array of 32-bit floating-point numbers representing the convolution kernel.
The output should be written to the output array, which will have a size of input_size - kernel_size + 1.

output[i] = foreach(j<kernel_size) { sum(input[i+j].kernel[j]) }

Implementation Requirements
Use only native features (external libraries are not permitted)
The solve function signature must remain unchanged
The final result must be stored in the array output
Example 1:
Input: input = [1, 2, 3, 4, 5], kernel = [1, 0, -1]
Output: [-2, -2, -2]
Example 2:
Input: input = [2, 4, 6, 8], kernel = [0.5, 0.2]
Output: [1.8, 3.2, 4.6]
Constraints
1 ≤ input_size ≤ 1,500,000
1 ≤ kernel_size ≤ 2047
kernel_size ≤ input_size
Performance is measured with input_size = 1,500,000, kernel_size = 2,047
*/

template <size_t ACCUM_PER_LANE>
__global__ void convolution_1d_kernel(const float* input, const float* kernel, float* output,
                                      int input_size, int kernel_size) {
    /* 
        - Each warp owns 32 outputs (32*4B = 128B coalesced store)
        - Each thread owns 1 of these outputs and reduces independently
        - ILP (for CUDA core utilization) and MLP (for SMEM latency hiding) is through warps/CTA
    */
    // Within all threadblocks/grid
    auto block_idx = blockIdx.x;
    auto blocks_in_grid = gridDim.x;
    
    // Within CTA
    auto warps_per_block = blockDim.x / 32;
    auto warp_idx = threadIdx.x / 32;
    auto lane_idx = threadIdx.x % 32;

    // Layout Indexing (output)
    uint8_t LANES_PER_WARP = 32;
    uint32_t OUTPUT_ELEMS_PER_WARP = 32 * ACCUM_PER_LANE;
    uint32_t output_start_idx = warp_idx * OUTPUT_ELEMS_PER_WARP + block_idx * warps_per_block * OUTPUT_ELEMS_PER_WARP;
    uint32_t output_step_idx = warps_per_block * blocks_in_grid * OUTPUT_ELEMS_PER_WARP;
    uint32_t output_size = input_size - kernel_size + 1;

    // Layout Indexing (input)
    uint32_t INPUT_ELEMS_OFFSET_PER_BLOCK = OUTPUT_ELEMS_PER_WARP * warps_per_block;
    uint32_t input_start_idx = INPUT_ELEMS_OFFSET_PER_BLOCK * block_idx;
    uint32_t input_step_idx = INPUT_ELEMS_OFFSET_PER_BLOCK * blocks_in_grid;

    // Layout indexing (smem)
    alignas(256) extern __shared__ float d_inputs[];
    uint32_t smem_load_start_idx = warp_idx * LANES_PER_WARP + lane_idx;
    uint32_t smem_load_step_idx = warps_per_block * LANES_PER_WARP;

    // Load Filter taps in shared memory
    __shared__ float d_kernel[2048];
    for(uint32_t idx = threadIdx.x; idx < kernel_size; idx += blockDim.x) {
        d_kernel[idx] = kernel[idx];
    }
    // Load inputs into shared memory
    uint32_t INPUT_ELEMS_PER_BLOCK = kernel_size + OUTPUT_ELEMS_PER_WARP * warps_per_block; // 1 element padded
    for (uint16_t idx = smem_load_start_idx; idx < INPUT_ELEMS_PER_BLOCK; idx += smem_load_step_idx) {
        // predicated load
        uint32_t global_input_idx = input_start_idx + idx;
        d_inputs[idx] = global_input_idx < input_size ? input[global_input_idx] : 0.0f;
    }
    __syncthreads(); // we rely on CTA-parallelism per SM to hide this (GMEM MLP)

    // outerloop
    for (uint32_t idx = output_start_idx; idx < output_size; idx += output_step_idx) {
        // calculate OUTPUT_ELEMS_PER_WARP = 32, 1 output per lane
        uint32_t filter_idx = 0;
        uint32_t warp_input_base = warp_idx * OUTPUT_ELEMS_PER_WARP;
        float output_reg[ACCUM_PER_LANE];
        float input_reg[ACCUM_PER_LANE];
#pragma unroll
        for (int acc = 0; acc < ACCUM_PER_LANE; acc++) {
            uint32_t acc_output_offset = acc * LANES_PER_WARP + lane_idx;
            output_reg[acc] = 0.0f;
            input_reg[acc] = d_inputs[warp_input_base + acc_output_offset];
        }
        for(; filter_idx < kernel_size; filter_idx++) {
            // load filter tap for all lanes of warp
            float filter_tap = d_kernel[filter_idx]; // broadcast to all threads
#pragma unroll
            for (int acc = 0; acc < ACCUM_PER_LANE; acc++) {
                uint32_t acc_output_offset = acc * LANES_PER_WARP + lane_idx;
                // FMA
                output_reg[acc] = output_reg[acc] + input_reg[acc] * filter_tap;
                // load next input
                input_reg[acc] = d_inputs[warp_input_base + acc_output_offset + filter_idx + 1];
            }
        }
#pragma unroll
        for (int acc = 0; acc < ACCUM_PER_LANE; acc++) {
            uint32_t acc_output_offset = acc * LANES_PER_WARP + lane_idx;
            uint32_t output_idx = idx + acc_output_offset;
            if (output_idx < output_size) {
                output[output_idx] = output_reg[acc];
            }
        }
    }
}

// input, kernel, output are device pointers (i.e. pointers to memory on the GPU)
extern "C" void solve(const float* input, const float* kernel, float* output, int input_size,
                      int kernel_size) {
    int output_size = input_size - kernel_size + 1;
    constexpr int accumPerLane = 4;
    int threadsPerBlock = 32 * 4;
    int outputsPerBlock = threadsPerBlock * accumPerLane;
    int blocksPerGrid = (output_size + outputsPerBlock - 1) / outputsPerBlock;
    size_t smem_bytes = (kernel_size + outputsPerBlock) * sizeof(float);

    convolution_1d_kernel<accumPerLane><<<blocksPerGrid, threadsPerBlock, smem_bytes>>>(input, kernel, output, input_size,
                                                              kernel_size);
    cudaDeviceSynchronize();
}
