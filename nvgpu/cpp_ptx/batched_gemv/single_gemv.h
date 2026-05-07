#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>
#include <cstdio>
#include <cstdint>
#include <type_traits>

#include "Utils.h"

template <size_t kUnroll = 2, kUnrollAlu = 2, size_t K = 128 /*hidden-dim*/, size_t kWarpTileN = 64 /* for coalesced writes for C */>
__global__ __launch_bounds__(128) gemv(
    int N,
    const __half* g_A, // [1, K]
    const __half* g_B, // [N, K]
    half* C,
)
{
    /* Currently the kernel is written for K=128 */
    static_assert(K == 128);
    static_assert(kWarpTileN == 64);

    __half2 r_A[K/64];
    // Each warp loads kWarpTileN rows to
    //  a.) write kWarpTileN * 2B in a coalesced manner to output
    //  b.) effectively acts as latency hiding component
    __half2 r_B[kUnroll][kWarpTileN][K/64];
    __half2 r_C[kWarpTileN/32 /* each lane is involved in writing out C */][K/64];
    // Q: Do I need to hide store latency??

    // Fast-reg lookups
    auto warp_idx = threadIdx.x / 32;
    auto cta_idx = blockIdx.x;
    auto num_warps = blockDim.x / 32;
    auto num_ctas = griDim.x;
    auto lane_idx = threadIdx.x & 31;
    
    // Single 512B = 16Bx32 load for EACH warp
    // need to cast to 64-bit format to load 8B per lane
    r_A = static_cast<const int2*>(g_A + lane_idx * 4);

    // Tensor movement
    uint32_t ROW_SIZE = K;
    uint32_t WARP_TILE_SIZE = kWarpTileN * row_size;
    uint32_t CTA_TILE_SIZE = num_warps * WARP_TILE_SIZE;

    auto g_Bcurr = g_B + cta_idx * CTA_TILE_SIZE;
    auto g_Ccurr = g_C + cta_idx * num_warps * kWarpTileN;
    auto g_Bstep = num_ctas * CTA_TILE_SIZE;
    auto g_Cstep = num_ctas * num_warps * kWarpTileN;

    /* ---- Prologue ---- */
    // For hiding GMEM Latency
    // and for ensuring coalesced stores despite tensor contraction
    // kWarpTileN * kUnroll rows will be prefetched
#pragma unroll
    for (auto unr = 0; unr < kUnroll; unr++) {
#pragma unroll 
        for (auto n = 0; n < kWarpTileN; n++) {
            auto g_Bload = g_Bcurr + (
                + warp_idx * WARP_TILE_SIZE
                + n * ROW_SIZE
            )
            r_B[unr][n] = static_cast<const int2*>(g_Bload + lane_idx * 4)
        }
        g_Bcurr += g_Bstep
    }

    g_Bstart += kUnroll * CTA_TILE_SIZE

    uint32_t rd_ptr = 0;
    uint32_t wr_ptr = 0;
    /* ----- Main Loop ----- */
    for (; g_Bcurr < (g_B + N); g_Bcurr += g_Bstep) {
        /* compute */
#pragma unroll
        for (auto n = 0; n < kWarpTileN; n += kUnrollAlu) {
            // Each iteration we compute on fp16x2 elements per instruction
            __half2 out[kUnrollAlu][2];

            // Packed float2 multiplies
#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {
                // iter[i] = A[1,K] @ B[K,1] -> C[i], C[i+1]
                out[unr][0] = __hmul(r_A[0], r_B[rd_ptr][n + unr][0]);
                out[unr][1] = __hmul(r_A[1], r_B[rd_ptr][n + unr][1]);
            }

            // fully unrolled bfly shuffle(s) for independent rows
#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 16);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 16);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 8);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 8);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 4);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 4);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 2);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 2);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 1);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 1);
            }
                
#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) { 
                // kWarpTileN == 64 for this to be possible.
                // needs to be masked/predicated instruction
                if (((n + unr) / 2) == lane_idx) {
                    r_C[(n + unr) % 2][0] = out[unr][0];
                    r_C[(n + unr) % 2][1] = out[unr][1];
                }
            }
        }
        
        /* store */
        // fully coalesced 128B / 64 elements per warp iff kWarpTileN = 128
        __half2 dpr_val = {r_C[0].x + r_C[0].y, r_C[1].x + r_C[1].y};
        *(reinterpret_cast<int2*>(g_Ccurr + warp_idx * kWarpTileN + lane_idx * 4)) = *(reinterpret_cast<int2*>(&dpr_val));

        /* load next CTA tile */
#pragma unroll 
        for (auto n = 0; n < kWarpTileN; n++) {
            auto g_Bload = g_Bcurr + (
                + warp_idx * WARP_TILE_SIZE
                + n * ROW_SIZE
            )
            r_B[wr_ptr][n] = static_cast<const int2*>(g_Bload + lane_idx * 4)
        }
        g_Bcurr += g_Bstep;
        g_Ccurr += g_Cstep;
        rd_ptr = (rd_ptr + 1) % kUnroll;
        wr_ptr = (wr_ptr + 1) % kUnroll;
    }

    /* ----- Epilogue ----- */
    for (auto unr = 1; unr < kUnroll; unr++) {
        /* compute and flush remaining CTA tiles */

        /* compute */
#pragma unroll
        for (auto n = 0; n < kWarpTileN; n++) {
            // Each iteration we compute on fp16x2 elements per instruction
            __half2 out[kUnrollAlu][2];

            // Packed float2 multiplies
#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {
                // iter[i] = A[1,K] @ B[K,1] -> C[i], C[i+1]
                out[unr][0] = __hmul(r_A[0], r_B[rd_ptr][n + unr][0]);
                out[unr][1] = __hmul(r_A[1], r_B[rd_ptr][n + unr][1]);
            }

            // fully unrolled bfly shuffle(s) for independent rows
#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 16);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 16);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 8);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 8);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 4);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 4);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 2);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 2);
            }

#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) {            
                out[unr][0] += __shfl_xor_sync(0xffffffffu, out[unr][0], 1);
                out[unr][1] += __shfl_xor_sync(0xffffffffu, out[unr][1], 1);
            }
                
#pragma unroll
            for (auto unr = 0; unr < kUnrollAlu; unr++) { 
                // kWarpTileN == 64 for this to be possible.
                // needs to be masked/predicated instruction
                if (((n + unr) / 2) == lane_idx) {
                    r_C[(n + unr) % 2][0] = out[unr][0];
                    r_C[(n + unr) % 2][1] = out[unr][1];
                }
            }
        }
        /* store */
        *(reinterpret_cast<int2*>(g_Ccurr + lane_idx * 4)) = *(reinterpret_cast<int2*>(r_C));
        g_Ccurr += g_Cstep;
        rd_ptr = (rd_ptr + 1) % kUnroll;
    }
}