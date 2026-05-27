#include <cuda_runtime.h>

template <size_t NUM_WARPS_PER_BLOCK>
__global__ __launch_bounds(1024) void reduction_kernel(const float* input, float* output, int N) {
    // Kernel Invariants
    int NUM_ELEMS_PER_THREAD = 2;

    // RMEM
    float2 thread_val = {0.0f, 0.0f};
    float sum_val = 0.0f;
    auto warp_idx = threadIdx.x / 32;
    auto lane_idx = threadIdx.x % 32;

    // SMEM
    __shared__ float block_val[NUM_WARPS_PER_BLOCK];

    if (threadIdx.x == 0) {
#pragma unroll
        for(int warp = 0; warp < NUM_WARPS_PER_BLOCK; warp++) {
            block_val[warp] = 0.0f;
        }
    }
    __syncthreads();

    // Load to RMEM
    int input_offset = (blockIdx.x * blockDim.x + threadIdx.x) * NUM_ELEMS_PER_THREAD;
    if (input_offset < N - 1) { // conditional load
        thread_val = *(reinterpret_cast<const float2*>(input + input_offset));
        sum_val = thread_val.x + thread_val.y;
    }
    else if (input_offset == N - 1) {
        sum_val = *(input + input_offset);
    }

    // Warp-Reduce - threads masked out have zeros
    for (int offset = 16; offset >= 1; offset = offset >> 1) {
        sum_val += __shfl_down_sync(0xffffffffu, sum_val, offset);
    }

    // Thread0 has final value write to SMEM
    if (lane_idx == 0) {
        block_val[warp] = sum_val;
    }

    // Wait for all warps to complete SMEM writes
    __syncthreads();

    // Final block-reduce and atomic-write to global sum
    sum_val = 0.0f;
    if (threadIdx.x == 0) {
#pragma unroll
        for(int warp = 0; warp < NUM_WARPS_PER_BLOCK; warp++) {
            sum_val += block_val[warp];
        }    
        atomicAdd(output, sum_val);
    }

}

// input, output are device pointers
extern "C" void solve(const float* input, float* output, int N) {
    cudaMemset(output, 0, sizeof(float));
    int NUM_ELEMS_PER_THREAD = 2; // fp32x2 packed reduce
    int NUM_THREADS_PER_BLOCK = 128; // 4 warps
    int NUM_ELEMS_PER_BLOCK = NUM_ELEMS_PER_THREAD * NUM_THREADS_PER_BLOCK;
    int NUM_BLOCKS = (N + NUM_ELEMS_PER_BLOCK - 1) / NUM_ELEMS_PER_BLOCK;

    dim3 grid_dim = NUM_BLOCKS;
    dim3 block_dim = NUM_THREADS_PER_BLOCK;
    reduction_kernel<<<grid_dim, block_dim>>>(input, output, N);
}
