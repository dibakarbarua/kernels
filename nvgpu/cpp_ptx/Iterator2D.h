#pragma once
#include <cuda.h>
#include <cassert>
#include <cstdint>
#include <cstddef>

// A simple 2D iterator for 2D tensors on GPUs
/*
For a given 2D tensor traversal,     
each iteration requires:
    - GMEM: 
        - xidx
        - yidx
        - tile_xsize
        - tile_ysize
    - SMEM:
        - tile_xsize
        - tile_ysize
global metadata:
    - tensor_stride_x
    - tensor_stride_y (only for transpose)
    - tensor_xsize
    - tensor_ysize
    - number of workers (PEs/CTAs)
*/

struct Tile2D {
    uint32_t xidx;
    uint32_t yidx;
};

template <size_t kNumWorkers>
class Iterator2D
{
    static_assert(kNumWorkers > 0, "Iterator2D requires at least one worker.");

    public:
        Iterator2D() = delete;
        __device__ __forceinline__ Iterator2D(uint32_t worker_idx, uint32_t tsr_xsize, uint32_t tsr_ysize, uint32_t tile_xsize, uint32_t tile_ysize);
        __device__ __forceinline__ Tile2D next();
        __device__ __forceinline__ bool end();
    private:
        uint32_t curr_tile_idx;
        uint32_t curr_tile_xidx;
        uint32_t curr_tile_yidx;
        uint32_t num_tiles_x;
        uint32_t num_tiles_y;
        uint32_t total_tiles;
        uint32_t tile_xsize;
        uint32_t tile_ysize;
        bool tiles_done;
        __device__ __forceinline__ void decode_current_tile();
        __device__ __forceinline__ void set_next_tile();
};

template <size_t kNumWorkers>
__device__ __forceinline__ Iterator2D<kNumWorkers>::Iterator2D(uint32_t worker_idx, uint32_t tsr_xsize, uint32_t tsr_ysize, uint32_t tile_xsize, uint32_t tile_ysize)
{
    // This is a linearized iterator, kernel should launch using a 1D grid.
    assert(blockDim.y == 1);
    assert(blockDim.z == 1);
    assert(gridDim.y == 1);
    assert(gridDim.z == 1);
    assert(tile_xsize > 0);
    assert(tile_ysize > 0);

    uint32_t const raw_num_tiles_x = (tsr_xsize + tile_xsize - 1) / tile_xsize;
    this->num_tiles_y = (tsr_ysize + tile_ysize - 1) / tile_ysize;
    this->num_tiles_x = raw_num_tiles_x > 0 ? raw_num_tiles_x : 1;
    this->total_tiles = raw_num_tiles_x * this->num_tiles_y;
    this->tile_xsize = tile_xsize;
    this->tile_ysize = tile_ysize;
    this->curr_tile_idx = worker_idx;
    this->decode_current_tile();
}

/*
CUTLASS-style FastDivMod
Since num_tiles_x is a runtime value, the two quotient and remainder will be expensive as ...
... the compiler will emit guards to protect against divide by zero etc.

To make this commonly used div-mod operation performant, CUTLASS exposes ...
... a FastDivMod operation that modifies the calculation to:

Let N = dividend (tile_idx) and d = divisor (num_tiles_x)
    p = 31u + ceil_log2(d)
    m = ceil(2^p / d)
      = (uint32_t(p << 31) + d - 1) / d
    shift = p - 32

The run-time calculations become:
q = (d == 1) ? (__umulhi(n,m) >> shift) : d
r = n - q * d

Q: Why __umulhi() ?
    - d >= 1
    - Hence p >= 32
    - Hence n * m = n * (>= 2^32) = n << 32
    - We are guaranteed that the lower 32-bits will be zero and the product will be 64-bit
    - On NVIDIA GPUs, ALUs are 32-bit ....

m and shift can be precalculated in the host using static constexpr
*/
template <size_t kNumWorkers, size_t m, size_t shift>
__device__ __forceinline__ void Iterator2D<kNumWorkers>::decode_current_tile()
{
    this->curr_tile_yidx = (this->num_tiles_x == 1) ? ((uint32_t)__umulhi(this->curr_tile_idx, m) >> shift) : this->num_tiles_x;
    this->curr_tile_xidx = this->curr_tile_idx - this->curr_tile_yidx * this->num_tiles_x;
    this->tiles_done = this->curr_tile_idx >= this->total_tiles;
}

template <size_t kNumWorkers>
__device__ __forceinline__ void Iterator2D<kNumWorkers>::set_next_tile()
{
    this->curr_tile_idx += kNumWorkers;
    this->decode_current_tile();
}

template <size_t kNumWorkers>
__device__ __forceinline__ bool Iterator2D<kNumWorkers>::end()
{
    return this->tiles_done;
}

template <size_t kNumWorkers>
__device__ __forceinline__ Tile2D Iterator2D<kNumWorkers>::next()
{
    assert(!this->tiles_done);
    Tile2D tile{this->curr_tile_xidx * this->tile_xsize,
                this->curr_tile_yidx * this->tile_ysize};
    this->set_next_tile();
    return tile;
}
