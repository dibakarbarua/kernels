#pragma once

#include <cstddef>
#include <cuda.h>
/*
--- Rotary Embedding (RoPE) application CUDA Kernel ---
Inputs:
    Q[Qh, S, D]
    K[Hk, S, D]
    cosines[S, D/2]
    sines[S, D/2]
Outputs:
    Q[S, D]
    K[S, D]
Algorithm:
    [ x0 ] * cos(0)        + [- x1 ] * sin(0)
    [ x1 ] * cos(0)        + [  x0 ] * sin(0)
    [ x2 ] * cos(1)        + [- x3 ] * sin(1)
    [ x3 ] * cos(1)        + [  x2 ] * sin(1)
    [ x4 ] * cos(2)        + [- x5 ] * sin(2)
    [ x5 ] * cos(2)        + [  x4 ] * sin(2)
....
    [xd-1] * cos(d/2)      + [- xd ] * sin(d/2)
    [xd  ] * cos(d/2)      + [ xd-1 ] * sin(d/2)

Work Unit = D * 2 + D/2 * 2 = 3 * D

Let's say D = 128, D/2 = 64,
    One D-dim can be computed by one warp or one SIMD engine

- Each warp can work on separate tokens but processes all heads (Qh + Kh) in sequence
- So NUM_TOTAL_TOKENS = S since 1 token * (Qh + Kh) constitutes one full token iteration for a warp.

So if NUM_TOKENS_PER_THREAD is our tunable,
    TOTAL_ITERATIONS_PER_THREAD = NUM_TOTAL_TOKENS / NUM_TOKENS_PER_THREAD

For each iteration we will do:
    - Load Q-tokens (Qh * NUM_TOKENS_PER_THREAD)
    - Load K-tokens (Kh * NUM_TOKENS_PER_THREAD)
    - Load cosines (NUM_TOKENS_PER_THREAD)
    - Load sines (NUM_TOKENS_PER_THREAD)
    - Compute and Store Q-tokens (Qh * NUM_TOKENS_PER_THREAD)
    - Compute and Store K-tokens (Kh * NUM_TOKENS_PER_THREAD)
*/

template <size_t NUM_TOKENS_PER_THREAD, size_t D>
__global__ void apply_rotary_emb(
    const half* Qin,  // [S, Qh, D]
    const half* Kin,  // [S, Kh, D]
    half* Qout,       // [S, Qh, D]
    half* Kout,       // [S, Kh, D]
    float* cosines, // [S, D/2]
    float* sines,   // [S, D/2]
    uint32_t Qh,
    uint32_t Kh,
    uint32_t S
)
{
    // Workload sizing
    // assert that no y-division partitions
    auto GRID_DIM = gridDim.x;
    auto block_idx = blockIdx.x;
    auto BLOCK_DIM = blockDim.x;
    uint32_t const WARP_DIM = 32;
    auto NUM_WARPS = BLOCK_DIM / WARP_DIM;
    auto tidx = threadIdx.x;
    auto widx = threadIdx.x / WARP_DIM;
    auto lane_idx = threadIdx.x % WARP_DIM;
    static constexpr uint32_t NUM_TOKENS_PER_BLOCK = NUM_TOKENS_PER_THREAD * 4;

    uint32_t token_start_idx = block_idx * NUM_TOKENS_PER_BLOCK + widx * NUM_TOKENS_PER_THREAD;
    uint32_t token_end_idx = S;
    uint32_t token_step_idx = GRID_DIM * NUM_TOKENS_PER_BLOCK + widx * NUM_TOKENS_PER_THREAD;

    // We are keeping r_cos and r_sin in registers instead of SMEM
    // The reason for this is even though are duplicating data across warps (4x), 
    //      we are not paying the 32-cycle SMEM latency on every unrolled iteration.
    float2 r_cos[NUM_TOKENS_PER_THREAD][D/4];
    float2 r_sin[NUM_TOKENS_PER_THREAD][D/4];

    for(uint32_t token_idx = token_start_idx; token_idx < token_end_idx; token_idx += token_step_idx) 
    {
#pragma unroll
        for(uint32_t token = token_idx; token < token_idx + NUM_TOKENS_PER_THREAD; token += 1) {
            // load sine, cosine values
            r_cos[token - token_idx][lane_idx] = reinterpret_cast<float2 const*>(cosines)[token + lane_idx];
            r_sin[token - token_idx][lane_idx] = reinterpret_cast<float2 const*>(sines)[token + lane_idx];
        }

        // Q-tokens
        for (uint32_t head_idx = 0; head_idx < Qh; head_idx++) {
            half2 q_tokens[NUM_TOKENS_PER_THREAD][D/2];

#pragma unroll
            for(uint32_t token = token_idx; token < token_idx + NUM_TOKENS_PER_THREAD; token += 1) {
                // load Q-tokens - 2 loads for 4 elements
                q_tokens[token - token_idx][lane_idx] = reinterpret_cast<half2 const*>(Q)[token + head_idx * S + lane_idx];
                q_tokens[token - token_idx][lane_idx + 1] = reinterpret_cast<half2 const*>(Q)[token + head_idx * S + lane_idx + 1];
            }

#pragma unroll
            for(uint32_t token = token_idx; token < token_idx + NUM_TOKENS_PER_THREAD; token += 1) {
                // compute RoPE for each token
                float r_cos0 = r_cos[token - token_idx][lane_idx].x;
                float r_cos1 = r_cos[token - token_idx][lane_idx].y;
                float r_sin0 = r_sin[token - token_idx][lane_idx + 1].x;
                float r_sin1 = r_sin[token - token_idx][lane_idx + 1].y;
                half q0 = q_tokens[token - token_idx][lane_idx].x;
                half q1 = q_tokens[token - token_idx][lane_idx].y;
                half q2 = q_tokens[token - token_idx][lane_idx + 1].x;
                half q3 = q_tokens[token - token_idx][lane_idx + 1].y;
                half qout0 = q0 * r_cos0 - q1 * r_sin0;
                half qout1 = q1 * r_cos0 + q0 * r_sin0;
                half qout2 = q2 * r_cos1 - q3 * r_sin1;
                half qout3 = q3 * r_cos1 + q2 * r_sin1;
                *(reinterpret_cast<half2*>(Qout + token + head_idx * S + lane_idx)) = {qout0, qout1};
                *(reinterpret_cast<half2*>(Qout + token + head_idx * S + lane_idx + 1)) = {qout2, qout3};
            }
        } // q-tokens complete

        // K-tokens
        for (uint32_t head_idx = 0; head_idx < Kh; head_idx++) {
            half2 k_tokens[NUM_TOKENS_PER_THREAD][D/2];

#pragma unroll
            for(uint32_t token = token_idx; token < token_idx + NUM_TOKENS_PER_THREAD; token += 1) {
                // load K-tokens - 2 loads for 4 elements
                k_tokens[token - token_idx][lane_idx] = reinterpret_cast<half2 const*>(K)[token + head_idx * S + lane_idx];
                k_tokens[token - token_idx][lane_idx + 1] = reinterpret_cast<half2 const*>(K)[token + head_idx * S + lane_idx + 1];
            }

#pragma unroll
            for(uint32_t token = token_idx; token < token_idx + NUM_TOKENS_PER_THREAD; token += 1) {
                // compute RoPE for each token
                float r_cos0 = r_cos[token - token_idx][lane_idx].x;
                float r_cos1 = r_cos[token - token_idx][lane_idx].y;
                float r_sin0 = r_sin[token - token_idx][lane_idx + 1].x;
                float r_sin1 = r_sin[token - token_idx][lane_idx + 1].y;
                half k0 = k_tokens[token - token_idx][lane_idx].x;
                half k1 = k_tokens[token - token_idx][lane_idx].y;
                half k2 = k_tokens[token - token_idx][lane_idx + 1].x;
                half k3 = k_tokens[token - token_idx][lane_idx + 1].y;
                half kout0 = k0 * r_cos0 - k1 * r_sin0;
                half kout1 = k1 * r_cos0 + k0 * r_sin0;
                half kout2 = k2 * r_cos1 - k3 * r_sin1;
                half kout3 = k3 * r_cos1 + k2 * r_sin1;
                *(reinterpret_cast<half2*>(Kout + token + head_idx * S + lane_idx)) = {kout0, kout1};
                *(reinterpret_cast<half2*>(Kout + token + head_idx * S + lane_idx + 1)) = {kout2, kout3};
            }
        } // k-tokens complete
    }
}