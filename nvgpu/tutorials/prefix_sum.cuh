#pragma once

#include <cstddef>
#include <cstdint>
#include <cuda.h>

/*
--- Segmented Scan Kernel for Prefix Sum ---
- The scan kernel performs a recursive operation on the previous-index's result and itself

[x0, x1, x2, x3 ...... xN]
-> scan
[x0, x0 . x1, x0 . x1. x2, x0 . x1 . x3, ....... , x0 . x1 . x2 . x3 ..... xN]

- The work division approaches for Scan employ the Kogge Stone and Brett Kung reduction tree algorithms
Described in:
- [x] (https://www.youtube.com/watch?v=VLdm3bV4bKo)
- [x] (https://www.youtube.com/watch?v=ZKrWyEqqPVY)

The most performant method for a very-very large array is a Segmented Scan Algorithm.
Note: ARG_IN_CAPS is a tuning parameter (kernel-template-param).

Kernel Interface:
    Input: input_;
    Output: output_, global_prefix_sum_;

1. Within a thread-block
- Each thread works on a sequence of elements (thread-coarsening) individually to allow barrier-free computation
    - During the computation, each thread maintains a prefix-sum accumulated so far
Registers per-thread:
    local_elements[ELEMS_PER_THREAD];
    local_prefix_sum;
Shared memory per-block:
    input_elements[ELEMS_PER_THREAD * BLOCK_DIM * STAGES];
    prefix_sum[BLOCK_DIM];

- After each thread is done with its sequence, all threads contribute their prefix-sums to a shared block
All threads in a block will synchronize at this point

- If block-size if BLOCK_DIM, we get prefix_sum[BLOCK_DIM]
- Now all threads need to scan across the prefix_sums to get correct correct prefix-sum for each block_idx
- In-place recursion requires double-buffering to avoid synchronization
    - prefix_sum[thread_idx = 0] = prefix_sum[0]
    - prefix_sum[thread_idx = 1] = prefix_sum[0] + prefix_sum[1]
    - prefix_sum[thread_idx = 2] = prefix_sum[0] + prefix_sum[1] + prefix_sum[2]
    ....
    - prefix_sum[block_idx = BLOCK_DIM-1] = prefix_sum[0] + prefix_sum[1] .... prefix_sum[BLOCK_DIM-1]

After completing the reduction-tree scan for each block, threads will synchronize again

- At this point, we will have the correct prefix_sum for each thread's elements
- We will now add the prefix_sum of each thread to it's scanned elements
    - local_elements[i] += prefix_sum[thread_idx = tidx]

- At this point we will also have the global_prefix_sum_ value (in tidx == BLOCK_DIM-1, elements[ELEMS_PER_THREAD-1])

2. Across-threadblocks
- Once each block has scanned it's elements and calculated it's own global_prefix_sum_,
    - each thread block needs to add it's global_prefix_sum_ to all prev blocks' global_prefix_sum_
    - once the final_prefix_sum is calculated, each thread can add it to it's elements again and write it out
- For this to work, each thread-block has to wait for all previous thread-blocks to finish

Let number of thread-blocks = GRID_DIM
Global Memory:
    global_prefix_sum_ [GRID_DIM];
    prev_block_ready_ [GRID_DIM]; -> to notify next block that sum is ready (spin-lock for next block)

Only one thread can perform spin-lock while other's wait at barrier
    final_prefix_sum += global_prefix_sum_[block_idx - 1];
- At this point each thread can add final_prefix_sum to it's individual elements and write-out the data

3. Staging Data to hide initial memory latency
TBD .....
*/

template <typename T, size_t ELEMS_PER_THREAD, size_t BLOCK_DIM = 256, size_t STAGES_PER_WARP = 1>
__global__ void scan_kernel(
    T* const g_input,
    T* g_output,
    volatile T* g_prefix_sum,
    volatile bool* g_block_ready
) {
    static_assert(BLOCK_DIM > 0);
    static_assert(ELEMS_PER_THREAD > 0);

    // Workload sizing
    // assert that no y-division partitions
    auto block_idx = blockIdx.x;
    auto tidx = threadIdx.x;
    if (blockDim.x != BLOCK_DIM) {
        return;
    }
    constexpr uint32_t BLOCK_ELEMS = BLOCK_DIM * ELEMS_PER_THREAD;
    T* const g_input_block = g_input + BLOCK_ELEMS * block_idx;
    T* g_output_block = g_output + BLOCK_ELEMS * block_idx;
    
    // Registers
    T r_elements[ELEMS_PER_THREAD];
    T r_prefix_sum = 0;

    // Shared Memory
    // 128B aligned so each SMEM container starts at subbank0
    __shared__ alignas(128) T s_elements[ELEMS_PER_THREAD * BLOCK_DIM];
    __shared__ alignas(128) T s_prefix_sum_buf0[BLOCK_DIM];
    __shared__ alignas(128) T s_prefix_sum_buf1[BLOCK_DIM];
    __shared__ alignas(128) T s_block_offset;
    T* inBuf = s_prefix_sum_buf0;
    T* outBuf = s_prefix_sum_buf1;

    // GMEM -> SMEM, elements
    // TODO: GMEM Latency is exposed here. STAGES??
    for (uint32_t eidx = tidx; eidx < BLOCK_ELEMS; eidx += BLOCK_DIM) {
        // vectorized across warp
        s_elements[eidx] = g_input_block[eidx];
    }
    __syncthreads(); // data-load from GMEM complete

    // Thread-Local Scan
    /*
    ----- Bank Conflicts -----
    - Each thread reads STRIDE = ELEMS_PER_THREAD * sizeof(T) bytes contiguously from SMEM
    - If STRIDE = 128B, we will completely serialize each thread's SMEM Load
    Define Swizzle:
    M = log2(sizeof(T))
    2^(M+S) = STRIDE
    S = log2(STRIDE) - M
    2^B = warp-size = 32, B = 5
    - NOT IMPLEMENTED. B,M,S will have to set up at host-side according to other tunables
    - Rd/Wr Swizzle Requirement for s_prefix_sum and s_elements will be different.
    */

    // Load all elements into registers first to hide SMEM latency
    // By unrolling this loop separately, the scheduler can burst these loads.
#pragma unroll
    for (uint32_t eidx = 0; eidx < ELEMS_PER_THREAD; ++eidx) {
        r_elements[eidx] = s_elements[tidx * ELEMS_PER_THREAD + eidx];
    }

    // Perform the local prefix sum in registers
#pragma unroll
    for(uint32_t eidx = 1; eidx < ELEMS_PER_THREAD; ++eidx) {
        r_elements[eidx] += r_elements[eidx - 1];
    }
    inBuf[tidx] = r_elements[ELEMS_PER_THREAD - 1];
    __syncthreads(); // thread-local scan complete

    // Kogge-Stone Reduction Tree across threads
    // TODO SMEM Latency is exposed here.
    // BLOCK_DIM >= 64 to hide SMEM latency!
    for(uint32_t stride = 1; stride < BLOCK_DIM; stride *= 2) {
        // Predicated loads and stores
        if (tidx >= stride) {
            outBuf[tidx] = inBuf[tidx] + inBuf[tidx - stride];
        } else {
            outBuf[tidx] = inBuf[tidx];
        }
        __syncthreads();
        // toggle ping and pong buffers
        T* temp = outBuf;
        outBuf = inBuf;
        inBuf = temp;
    }

    // Final Reduction for each thread's elements

    // Step1. One thread waits until the previous block's cumulative sum is ready,
    // then publishes that scalar through shared memory.
    if (tidx == 0) {
        s_block_offset = 0;
        if (block_idx > 0) {
            while(!g_block_ready[block_idx - 1]) {
            }
            s_block_offset = g_prefix_sum[block_idx - 1];
        }
        g_prefix_sum[block_idx] = s_block_offset + inBuf[BLOCK_DIM - 1];
        // Make the cumulative sum visible before the ready flag is observed.
        __threadfence();
        g_block_ready[block_idx] = true;
    }
    __syncthreads(); // All threads have arrived, now we can broadcast-add

    // Step2. Add previous-block and previous-thread offsets to local values.
    r_prefix_sum = s_block_offset;
    if (tidx > 0) {
        r_prefix_sum += inBuf[tidx - 1];
    }
#pragma unroll
    for (uint32_t eidx = 0; eidx < ELEMS_PER_THREAD; ++eidx) {
        r_elements[eidx] += r_prefix_sum;
    }

    // Step3. Stage results back to SMEM, then write out coalesced.
#pragma unroll
    for (uint32_t eidx = 0; eidx < ELEMS_PER_THREAD; ++eidx) {
        s_elements[tidx * ELEMS_PER_THREAD + eidx] = r_elements[eidx];
    }
    __syncthreads();
    for (uint32_t eidx = tidx; eidx < BLOCK_ELEMS; eidx += BLOCK_DIM) {
        g_output_block[eidx] = s_elements[eidx];
    }
}
