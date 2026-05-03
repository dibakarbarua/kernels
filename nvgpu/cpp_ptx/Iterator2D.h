#pragma once
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
    uint32_t xsize;
    uint32_t ysize;
};

template <typename T, size_t kNumWorkers>
class Iterator2D
{
    public:
        Iterator2D() = delete;
        Iterator2D(uint32_t worker_idx, uint32_t tsr_xsize, uint32_t tsr_ysize, uint32_t tile_xsize, uint32_t tile_ysize);
        void set_next_tile();
        Tile2D next();
        bool end();
    private:
        uint32_t tsr_xsize;
        uint32_t tsr_ysize;
        uint32_t tile_xsize;
        uint32_t tile_ysize;
        uint32_t curr_tile_idx;
        uint32_t curr_tile_xidx;
        uint32_t curr_tile_yidx;
        uint32_t curr_xidx;
        uint32_t curr_xsize;
        uint32_t curr_yidx;
        uint32_t curr_ysize;
        uint32_t num_tiles_x;
        uint32_t num_tiles_y
        uint32_t last_xsize;
        uint32_t last_ysize;
        bool tiles_done;
};

template <typename T, size_t kNumWorkers>
Iterator2D<T, kNumWorkers>::Iterator2D(uint32_t worker_idx, uint32_t tsr_xsize, uint32_t tsr_ysize, uint32_t tile_xsize, uint32_t tile_ysize)
{
    tsr_xsize = tsr_xsize;
    tsr_ysize = tsr_ysize;
    tile_xsize = tile_xsize;
    tile_ysize = tile_ysize;
    num_tiles_x = (tsr_xsize + xsize - 1) / xsize;
    last_xsize = tsr_xsize % xsize; // rem is one instruction in GPUs
    num_tiles_y = (tsr_ysize + ysize - 1) / ysize;
    last_ysize = tsr_ysize - (num_tiles_y - 1) * ysize;
    curr_tile_idx = 0;
    set_next_tile();
}

template<typename T, size_t kNumWorkers>
void Iterator2D<T, kNumWorkers>::set_next_tile()
{
    curr_tile_idx += worker_idx;
    curr_tile_yidx = curr_tile_idx / num_tiles_x;
    curr_tile_xidx = curr_tile_idx % num_tiles_x; // rem is one instruction in GPUs
    curr_xsize = (curr_tile_xidx == num_tiles_x - 1) ? tile_xsize : last_xsize;
    curr_ysize = (curr_tile_yidx == num_tiles_y - 1) ? tile_ysize : last_ysize;
    curr_xidx = curr_tile_xidx * tile_xsize;
    curr_yidx = curr_tile_yidx * tile_ysize;
    tiles_done = curr_tile_idx < num_tiles_x * num_tiles_y;
}

template<typename T, size_t kNumWorkers>
bool Iterator2D<T, kNumWorkers>::end()
{
    return tiles_done;
}

template<typename T, size_t kNumWorkers>
Tile2D Iterator2D<T, kNumWorkers>::next()
{
    assert(!tiles_done);
    Tile2D tile{curr_xidx, curr_yidx, curr_size, curr_ysize};
    set_next_tile();
    return tile;
}
