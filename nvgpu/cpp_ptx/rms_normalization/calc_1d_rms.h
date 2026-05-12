#include <cstdint>
#include <cuda_runtime.h>

template<std::size_t kNumIters, std::size_t kPackedElements, std::size_t kUnroll>
__global__ __launch_bounds__(1024) fp32_rms_norm_calculate_rms(const float* input, float gamma, float beta, float* output, int N,
                      float eps, alignedN) {
    int warp_idx = threadIdx.x / 32;
    int block_idx = blockIdx.x;
    int thread_idx = threadIdx.x % 32;
    int block_size = blockDim.x;
    int warp_size = 32;

    uint32_t input_start_idx = (block_idx * block_size + warp_idx * warp_size + thread_idx) * kPackedElements;
    uint32_t input_step_idx = gridDim.x * block_size * kPackedElements;

    // SMEM
    alignas(8) __shared__ float shared_sumOfSquares;
    
    if (thread_idx == 0) {
        shared_sumOfSquares = 0.0f;
    }
    __syncthreads();

    // RF
    float2 reg_inputs[kUnroll];
    float2 reg_sumOfsquares[kUnroll];

    // zero out sum of squares
#pragma unroll
    for(uint32_t idx = 0; idx < kUnroll; idx++) {
        reg_sumOfSquares = {0.0f, 0.0f};
    }

    // aligned-loop
    for(uint32_t idx = input_start_idx; idx < alignedN; idx += input_step_idx * kUnroll) {
        float2* input_addr = reinterpret_cast<float2*>(input + input_start_idx);
        reg_inputs[0] = *(reinterpret_cast<float2*>(input_addr + input_step_idx));
        reg_inputs[1] = *(reinterpret_cast<float2*>(input_addr + input_step_idx * 2));
        reg_inputs[2] = *(reinterpret_cast<float2*>(input_addr + input_step_idx * 3));
        reg_inputs[3] = *(reinterpret_cast<float2*>(input_addr + input_step_idx * 4));

        reg_sumOfSquares[0] = {
            reg_inputs[0].x * reg_inputs[0].x + reg_sumOfSquares[0].x,
            reg_inputs[0].y * reg_inputs[0].y + reg_sumOfSquares[0].y,
        };
        reg_sumOfSquares[1] = {
            reg_inputs[1].x * reg_inputs[1].x + reg_sumOfSquares[1].x,
            reg_inputs[1].y * reg_inputs[1].y + reg_sumOfSquares[1].y,
        };
        reg_sumOfSquares[2] = {
            reg_inputs[2].x * reg_inputs[2].x + reg_sumOfSquares[2].x,
            reg_inputs[2].y * reg_inputs[2].y + reg_sumOfSquares[2].y,
        };
        reg_sumOfSquares[2] = {
            reg_inputs[3].x * reg_inputs[3].x + reg_sumOfSquares[3].x,
            reg_inputs[3].y * reg_inputs[3].y + reg_sumOfSquares[3].y,
        };
    }

    // tail-loop
    input_start_idx += alignedN;
    for(uint32_t idx = input_start_idx; idx < N; idx += input_step_idx) {
        if (idx < N) {
            reg_inputs[0] = *(reinterpret_cast<float2*>(input_addr + input_step_idx));
            reg_sumOfSquares[0] = {
                reg_inputs[0].x * reg_inputs[0].x + reg_sumOfSquares[0].x,
                reg_inputs[0].y * reg_inputs[0].y + reg_sumOfSquares[0].y,
            };
        }
    }

    __syncwarp(); // entire warp has finished, reduce across warp

    for(uint32_t offset = 16; offset >= 1; offset >>= 1) {
        reg_sumOfSquares[0].x = __shfl_down_sync(offset, reg_sumOfSquares[0].x);
        reg_sumOfSquares[0].y = __shfl_down_sync(offset, reg_sumOfSquares[0].y);
        reg_sumOfSquares[1].x = __shfl_down_sync(offset, reg_sumOfSquares[1].x);
        reg_sumOfSquares[1].y = __shfl_down_sync(offset, reg_sumOfSquares[1].y);
        reg_sumOfSquares[2].x = __shfl_down_sync(offset, reg_sumOfSquares[2].x);
        reg_sumOfSquares[2].y = __shfl_down_sync(offset, reg_sumOfSquares[2].y);
        reg_sumOfSquares[3].x = __shfl_down_sync(offset, reg_sumOfSquares[3].x);
        reg_sumOfSquares[3].y = __shfl_down_sync(offset, reg_sumOfSquares[3].y);
    }

    // only lane0 of each warp has the correct reduced values in reg_sumOfSquares[];
    if (thread_idx == 0) {
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[0].x);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[0].y);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[1].x);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[1].y);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[2].x);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[2].y);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[3].x);
        atomicAdd(&shared_sumOfSquares, reg_sumOfSquares[3].y);
    }

    // CTA sumOfSquares is ready, synchronize
    __syncthreads();

    // At this point we can deploy a cluster-write to one CTA's SMEM and then put a cluster-barrier
    // This will reduce global memory contention

    if (thread_idx == 0) {
     atomicAdd(output, shared_sumOfSquares);
    }
}

// input, output are device pointers
extern "C" void solve(const float* input, float gamma, float beta, float* output, int N,
                      float eps) {
    static constexpr int NUM_ELEMS_PER_THREAD = 2; // fp32x2
    static constexpr int NUM_THREADS_PER_WARP = 32;
    static constexpr int BLOCK_SIZE = NUM_WARPS_PER_BLOCK * NUM_THREADS_PER_WARP * NUM_ELEMS_PER_THREAD; // threads
    static constexpr int ELEMENTS_PER_BLOCK = BLOCK_SIZE * NUM_ITERS;
    
    // Tunables
    static constexpr int NUM_WARPS_PER_BLOCK = 4; // Tune CTA size - SMEM/ALU latency hiding
    static constexpr int NUM_ITERS = 4; // Tune number of CTAs - GMEM latency hiding
    static constexpr int UNROLL = 4; // Tune ILP - SMEM/ALU latency hiding
    
    assert(NUM_ITERS >= UNROLL);
    assert(NUM_ITERS % UNROLL == 0);

    int alignedN = ((int) (N / ELEMENTS_PER_BLOCK) * ELEMENTS_PER_BLOCK );

    int GRID_SIZE = (N + ELEMENTS_PER_BLOCK - 1) / ELEMENTS_PER_BLOCK;
    fp32_rms_norm_calculate_rms<NUM_ITERS, NUM_ELEMS_PER_THREAD, UNROLL><<<GRID_SIZE, BLOCK_SIZE>>>(input, gamma, beta, output, N, eps, alignedN);

}
