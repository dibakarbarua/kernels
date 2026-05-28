#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace gemm_fma_ldst {

template <size_t bM, size_t bN, size_t bK,  // CTA-tile
          size_t wM, size_t wN,             // warp-tile
          size_t tM, size_t tN,             // thread-coarsening
          >          
__global__ __launch_bounds(1024) gemm(
    int M, int N, int K, half* A, half* B, float* C
)
{
    /*
    - The entire C-tile (MxN) is divided among CTAs into C-tiles (bM, bN)
    - The C-tile is accumulated over A and B tiles of bMxbK and bNxbK
    - The full C-tile is divided in an output stationary manner
        - 2 warps over bM, rest of the warps (2/4) over bN
        - Each warp processes a wM x wN portion of C-tile in this manner
        - Each warp sub-tile is tM x tN x 32 (tMxtN sub-tile per thread)
        - num_iters_M = wM/tM, num_iters_N = wN/(tN * 32)
    - To calculate one warp sub-tile of wM x wN, we have to work on AB tiles of tMxbK and tNx32xbK and accumulate into warp sub-tile
    - Once all warp-subtiles are done we can cooperatively write out the full CTA C-tile
        - We wait for all warp-subtiles to complete for warp-tiling (reuse)
    - LD/ST
        - Loads are into SMEM, Stores are from registers
        - Fully vectorized as float4 (16B) LD/ST
    */

    /* Thread Hierarchy */
    int WARPSIZE = 32;
    int BLOCKSIZE = blockDim.x / WARPSIZE; // 4 or 8
    int BLOCKSIZE_Y = 2; // 2 warps per bM
    int BLOCKSIZE_X = BLOCKSIZE / BLOCKSIZE_Y;
    auto lane_idx = threadIdx.x % WARPSIZE;
    auto warp_idx = threadIdx.x / WARPSIZE;
    int warp_idx_x = warp_idx % BLOCKSIZE_X;
    int warp_idx_y = warp_idx / BLOCKSIZE_X;
    half* A_ptr = A + blockIdx.y * bM * K; // assuming K-major
    half* B_ptr = B + blockIdx.x * bN * K; // assuming K-major
    half* C_ptr = C + blockIdx.y * N * bM + blockIdx.x * bN;

    /* Tiling */
    constexpr int num_iters_M = bM / tM;
    constexpr int num_iters_N = bN / (tN * WARPSIZE);
    int warp_col_start = wN * warp_idx_x;
    int warp_row_start = wM * warp_idx_y;
    
    /* Shared Memory */
    alignas(16) __shared__ sA[bM * bK];
    alignas(16) __shared__ sB[bN * bK];

    /* Registers */
    half2 rA[num_iters_m][tM / 2];
    half2 rB[num_iters_n][tN / 2];
    float2 rC[num_iters_m][num_iters_n][tM / 2][tN / 2] = {};

    /* Outer-loop over bK tiles for full K */
    for (int k = 0; k < K; k += bK) {
        /* Global Loads */
        // Load the full sA and sB in SMEM
        float4* gA_start = *reinterpret_cast<float4*>(A_ptr + warp_row_start * K + lane_idx * 4);
        float4* gB_start = *reinterpret_cast<float4*>(B_ptr + warp_col_start * K + lane_idx * 4);

        /* Compute */

        /* Stores */
    }


}
}
