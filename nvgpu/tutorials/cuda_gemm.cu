#include <cuda.h>
#include <cute/tensor.hpp>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

/* ---------------------------------- Naive GEMM --------------------------------------------- */

__global__ void __launch_bounds__(1024) naive_gemm_kernel(
    __restrict__ float* d_A,
    __restrict__ float* d_B,
    __restrict__ float* d_C,
    int M, int N, int K)
{
    // Thread Hierarchy
    uint16_t WARPSIZE = 32;
    auto num_threads_x = blockDim.x;
    auto num_blocks_x = gridDim.x;
    auto num_blocks_y = gridDim.y;
    auto lane_idx = threadIdx.x % WARPSIZE;

    // Tensor Indexing
    // A is row major, B is column major, C is row major
    uint32_t c_idx = blockIdx.y * N + blockIdx.x * num_threads_x + lane_idx;
    uint32_t a_row_idx = blockIdx.y; // broadcast to all threads in thread-block
    // Optimization 1:
    /*
        B is col-major, consecutive threads mapped to strided address
        stride = K
        We can coalesce this across threads by:
            - Using SMEM tiling with bK = 32 and bN = 32
            - bK = 32 allows us to load coalesced across a warp
            - bN = 32 allows us to store coalesced across a warp
    */
    uint32_t b_col_idx = blockIdx.x * num_threads_x + lane_idx;

    // Registers
    // Optimization 2:
    /*
        We can increase ILP and MLP and reduce instruction count by using thread-coarsening
            - This is the same as Register Tiling
            - Use wide-width load instructions (float2/float4)
            - Use packed-float ALU engines
            - Compute multiple C-values in one thread
    */
    float r_A = 0.0f;
    float r_B = 0.0f;
    float r_C = 0.0f;

    for (auto k = 0; k < K; k++) {
        r_A = d_A[a_row_idx * K + k];
        if (b_col_idx < N) {
            r_B = d_B[k * K + b_col_idx];
        }
        r_C += r_A * r_B;
    }
    if (b_col_idx < N) {
        d_C[c_idx] = r_C;
    }

    // Optimization 3:
    /*
        Total threads invoked = M * N
        Per-thread A loads = K
        Per-thread B loads = K
        Per-thread C stores = 1
        Per-thread FLOPS (non-indexing) = 2K
        Total FLOPS = 2MNK
        Total Data Movement = MNK + MNK + MN
        Arithmetic Intensity ~= 1
        We can improve this by doing:
            - Shared Memory Tiling
            - Register Tiling
    */
}

void naive_gemm_launcher(
    __restrict__ float* d_A,
    __restrict__ float* d_B,
    __restrict__ float* d_C,
    int M, int N, int K)
{
    // Tunables
    uint16_t threads_per_block = 256;
    
    // C is output stationary and mapped to different threads
    uint16_t WARPSIZE = 32;
    uint16_t num_threads_x = (N + 32 - 1) / 32;
    uint16_t num_blocks_x = (num_threads_x + threads_per_block - 1) / threads_per_block;
    uint16_t num_blocks_y = M;

    dim3 block_dim(num_threads_x, 1, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    naive_gemm_kernel<<<grid_dim, block_dim>>>(d_A, d_B, d_C, M, N, K);
}

/* ---------------------------------- Naive GEMM --------------------------------------------- */

/* ---------------------------------- SMEM Tiling GEMM --------------------------------------------- */

template <std::size_t bM, std::size_t bN, std::size_t bK>
__global__ void __launch_bounds__(1024) smem_tiling_gemm_kernel(
    __restrict__ float* d_A,
    __restrict__ float* d_B,
    __restrict__ float* d_C,
    int M, int N, int K)
{
    // Thread Hierarchy
    uint16_t WARPSIZE = 32;
    auto num_threads_x = blockDim.x;
    constexpr int num_warps = num_threads_x / 32;
    auto num_blocks_x = gridDim.x;
    auto num_blocks_y = gridDim.y;
    auto lane_idx = threadIdx.x % WARPSIZE;
    auto warp_idx = threadIdx.x / WARPSIZE;

    // Shared Memory
    __shared__ float s_A[bM * bK];
    __shared__ float s_B[bN * bK];

    // Registers
    constexpr int numAccM = (bM + num_warps - 1) / num_warps;
    constexpr int numAccN = bN;
    float r_C[numAccM][numAccN];

    // Initialize Accumulators
    uint32_t c_tile_ystart = warp_idx;
    uint32_t c_tile_ystep = num_warps;
    for(auto c_tile_yidx = c_tile_ystart; c_tile_yidx < bM; c_tile_yidx += c_tile_ystep) {
        for(auto c_tile_xidx = lane_idx; c_tile_xidx < bN; c_tile_xidx += WARPSIZE) {
            if ((c_tile_yidx < bM) && (c_tile_xidx < bN)) {
                r_C[c_tile_yidx][c_tile_xidx] = 0.0f
            }
        }
    }
    __syncthreads();

    // Outer-loop : K-tiles
    for(auto k_idx = 0; k_idx < K; k_idx += bK) {
        // 1.0 Load data into SMEM tile for A and B
        
        // Load A
        uint32_t a_row_start = blockIdx.y + warp_idx;
        uint32_t a_row_step = num_warps;
        uint32_t a_col_start = k_idx + lane_idx;
        uint32_t a_col_step = WARPSIZE;
        for(auto a_row_idx = a_row_start; a_row_idx < bM; a_row_idx += a_row_step) {
            for(auto a_col_idx = a_col_start; a_col_idx < bK; a_col_idx += a_col_step) {
                auto a_tile_yidx = a_row_idx - blockIdx.y;  // warp_idx
                auto a_tile_xidx = a_col_idx - k_idx;       // lane_idx
                if ((a_row_idx < bM) && (a_col_idx < bK) && (a_row_idx < M) && (r_col_idx < K)) {
                    s_A[a_tile_yidx * bK + a_tile_xidx] = d_A[a_row_idx * K + a_col_idx];
                }
            }
        }
        // Load B
        uint32_t b_col_start = blockIdx.x + warp_idx;
        uint32_t b_col_step = num_warps;
        uint32_t b_row_start = k_idx + lane_idx;
        uint32_t b_row_step = WARPSIZE;
        for(auto b_col_idx = b_col_start; b_col_idx < bN; b_col_idx += b_col_step) {
            for(auto b_row_idx = b_row_start; b_row_idx < bK; b_row_idx += b_row_step) {
                auto b_tile_xidx = b_row_idx - k_idx;
                auto b_tile_yidx = b_col_idx - blockIdx.x;
                if ((b_col_idx < bN) && (b_row_idx < bK) && (b_col_idx < N) && (b_row_idx < K)) {
                    s_B[b_tile_yidx * bK + b_tile_xidx] = d_B[b_row_idx * N + b_col_idx];
                }
            }
        }

        // Synchronize - Finish writing before next read
        __syncthreads();

        // 2.0 Compute r_C += s_A
        uint32_t c_tile_xidx = lane_idx;
        uint32_t c_tile_yidx = warp_idx;
        uint32_t c_tile_xstep = WARPSIZE;
        uint32_t c_tile_ystep = num_warps;
        uint32_t bK_idx_start = lane_idx;
        uint32_t bK_idx_step = WARPSIZE;
        for(; c_tile_yidx < numAccM; c_tile_yidx += c_tile_ystep) {
            for(; c_tile_xidx < numAccN; c_tile_xidx += c_tile_xstep) {
                for(auto bK_idx = bK_idx_start; bK_idx < bK; bK_idx += bK_idx_step) {
                    if ((c_tile_xidx < numAccN) && (c_tile_yidx < numAccM) && (bK_idx < bK)) {
                        r_C[c_tile_yidx][c_tile_xidx] += 
                            s_A[c_tile_yidx * bK + bK_idx] * s_B[c_tile_xidx * bK + bK_idx];
                    }
                }
            }
        }

        // Synchronize - Finish reading before next write
        __syncthreads();
    }
    // 3.0 Store Accumulator to d_C
    uint32_t c_tile_xidx = lane_idx;
    uint32_t c_tile_yidx = warp_idx;
    uint32_t c_tile_xstep = WARPSIZE;
    uint32_t c_tile_ystep = num_warps;
    uint32_t bK_idx_start = lane_idx;
    uint32_t bK_idx_step = WARPSIZE;
    for(; c_tile_yidx < numAccM; c_tile_yidx += c_tile_ystep) {
        for(; c_tile_xidx < numAccN; c_tile_xidx += c_tile_xstep) {
            uint32_t c_row_idx = blockIdx.y + c_tile_yidx;
            uint32_t c_col_idx = blockIdx.x + c_tile_xidx;
            if ((c_row_idx < M) && (c_col_idx < N)) {
                d_C[c_row_idx * M + c_col_idx] = r_C[c_tile_yidx][c_tile_xidx];
            }
        }
    }

    // Optimizations in SMEM Tiled GEMM:
    /*
        - Naive GEMM:
            Total threads invoked = M * N
            Per-thread A loads = K
            Per-thread B loads = K
            Per-thread C stores = 1
            Per-thread FLOPS (non-indexing) = 2K
            Total FLOPS = 2MNK
            Total Data Movement = MNK + MNK + MN
            Arithmetic Intensity ~= 1
            We can improve this by doing:
                - Shared Memory Tiling
                - Register Tiling
        - SMEM Tiled GEMM
            Per-SMEM-tile (bMxbK) A loads = bM*bK
            Per-SMEM-tile (bNxbK) B loads = bN*bK
            Per-SMEM-tile (bMxbN) C stores = bM*bN
            Per-SMEM-tile FLOPS = 2*bM*bN*K
            SMEM Tiles = MN/(bM * bN)
            Total FLOPS = 2MNK
            Total Data Movement = MN(bK/bN) + MN(bK/bM) + MN
            Arithmetic Intensity ~= 2K/(bK/bN + bK/bM + 1)
                [max = 2K for very small bK and very large bM/bN]
    */
}

void smem_tiling_gemm_launcher(
    __restrict__ float* d_A,
    __restrict__ float* d_B,
    __restrict__ float* d_C,
    int M, int N, int K)
{
    // Tunables
    uint16_t threads_per_block = 256;
    uint16_t bM = 32;
    uint16_t bN = 32;
    uint16_t bK = 32;

    // Static Checks
    uint16_t WARPSIZE = 32;
    uint16_t num_warps = threads_per_block / WARPSIZE;
    assert(bM >= num_warps); // coalesced A-loads (different row per warp)
    assert(bN >= num_warps); // coalesced B-loads (different row per warp)
    
    // C is output stationary and mapped to different threads
    uint16_t num_blocks_x = (N + bN - 1) / bN;
    uint16_t num_blocks_y = (M + bM - 1) / bM;

    dim3 block_dim(threads_per_block, 1, 1);
    dim3 grid_dim(num_blocks_x, num_blocks_y, 1);
    smem_tiling_gemm_kernel<<<grid_dim, block_dim>>><bM, bN, bK>(d_A, d_B, d_C, M, N, K);
}


/* ---------------------------------- SMEM Tiling GEMM --------------------------------------------- */

int main() {
    int M = 8192;
    int N = 8192;
    int K = 8192;

    thrust::host_vector<float> h_A(M * K);
    thrust::host_vector<float> h_B(N * K);
    thrust::host_vector<float> h_C(M * N);

    for(auto idx = 0; idx < M * K; idx++) {
        h_A[idx] = idx * 0.01;
    }
    for (auto idx = 0; idx < N * K; idx++) {
        h_B[idx] = idx * 0.02;
    }

    thrust::device_vector<float> d_A = h_A;
    thrust::device_vector<float> d_B = h_B;
    thrust::device_vector<float> d_C = h_C;

    int gemm_type = 0;

    if (gemm_type == 0) {
        naive_gemm(
            reinterpret_cast<float*>(thrust::raw_pointer_cast(d_A.data())),
            reinterpret_cast<float*>(thrust::raw_pointer_cast(d_B.data())),
            reinterpret_cast<float*>(thrust::raw_pointer_cast(d_C.data())),
            M, N, K
        );
    }

    thrust::copy(d_C.begin(), d_C.end(), h_C.begin());
    // h_C has final MxN tensor
}