#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "Utils.h"

template <typename T, size_t kUnroll, size_t K = 128 /*hidden-dim*/>
__global__ __launch_bounds__(512) gemv(
    int M,
    int N,
    const __half* g_A, // [1, K]
    const __half* g_B, // [K, N]
    half* C,
)
{
    /* Currently the kernel is written for K=128 */
    static_assert(K == 128);

    __half2 r_A[2];
    __half2 r_B[kUnroll][2];
    __half2 r_C[2];
    // Q: Do I need to hide store latency??

    // Fast-reg lookups
    auto warp_idx = threadIdx.x / 32;
    auto cta_idx = blockIdx.x;
    auto num_warps = blockDim.x;
    auto num_ctas = griDim.x;
    auto lane_idx = threadIdx.x & 31;
    
    // Single 512B = 16Bx32 load for EACH warp
    // need to cast to 64-bit format to load 8B per lane
    r_A = static_cast<const int2*>(g_A + lane_idx * 4);

    // Tensor movement
    uint32_t row_size = K * sizeof(__half);
    uint32_t warp_tile_size = num_warps * row_size;

    // Prologue for hiding GMEM Latency
#pragma unroll
    auto g_Bload = g_B;
    for (auto n = 0; n < kUnroll; n++) {
        g_Bload += ( 
            + cta_idx * kUnroll * num_warps * K * sizeof(__half) 
            + n * num_warps * K * sizeof(__half)
            + warp_idx * K * sizeof(__half)
        )
        r_B[n] = static_cast<const int*>(g_Bload + lane_idx * 4)
    }

    for (auto n = c)

}