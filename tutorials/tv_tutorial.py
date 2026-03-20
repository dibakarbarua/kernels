import torch
from cutlass.cute.runtime import from_dlpack
import cutlass.torch as cutlass_torch

def make_tensor(m, n, dtype):
    shape = (m, n)
    return (
        torch.empty(*shape, dtype=torch.int32)
        .random_(-2, 2)
        .to(dtype=dtype, device="cpu")
    )

@cute.jit
def kernel_tiling_examples(gmem_tsr: cute.Tensor):
    ###################################################
    #################### GMEM Tiling ##################
    ###################################################

    # row major 512x512 tensor
    gmem = cute.make_layout((512, 512), stride=(512, 1))
    
    # Let CTA Tile be 128x128
    cta_tile = (128, 128)

    tiled_gmem = cute.zipped_divide(gmem, tiler=cta_tile)
    cute.printf(">?? GMEM Layout: {}", gmem)
    cute.printf(">?? Tiled GMEM Layout: {}", tiled_gmem)

    # tiled_gmem has shape ((128,128), (4,4))
    # Now let's say we want to index into one of these 4x4 tiles

    # Slicing out one specific tile_idx
    cta_coord = (0, 0)
    local_tile = cute.local_tile(gmem_tsr, cta_tile, cta_coord, proj=(1,1))
    cute.printf(">?? Local Tile Layout: {}", local_tile.layout)

    # Slicing out a number of tile indices based on available threads/CTAs
    all_tiles = cute.zipped_divide(gmem_tsr, cta_tile)
    cute.printf(">?? All Tiles Layout: {}", all_tiles.layout)
    tiles = cute.size(all_tiles, mode=[1])
    cute.printf(">?? Number of Tiles: {}", tiles)

    ###########################################################################
    # I would have to loop through all of the tiles for a given CTA
    # This is usually a runtime artifact so static unroll would not be possible
    ###########################################################################

    ###########################################################################
    ######################### CTA Tile Partioning #############################
    ###########################################################################

    # In the example above a single CTA Tile is 128x128
    thread_coord = (0, 0)
    # horizontally stacked vertical warps
    thread_layout = cute.make_layout((32, 4), stride=(1, 32))
    value_layout = cute.make_layout((4, 32), stride=(1, 4))
    tiler_mn, tv_layout = cute.make_layout_tv(thread_layout, value_layout)
    tv_tsr = cute.composition(local_tile, tv_layout)

    cute.printf(">?? Thread Layout: {}", thread_layout)
    cute.printf(">?? Value Layout: {}", value_layout)
    cute.printf(">?? Tiler MN: {}", tiler_mn)
    cute.printf(">?? Thread-Value Layout: {}", tv_layout)
    cute.printf(">?? Thread-Value Tensor Layout: {}", tv_tsr.layout)
    
    tidx = (0, 1) # warp-1 lane-0
    tv_warp1_lane0 = tv_tsr.layout(tidx)
    cute.printf(">?? Value Index for warp 1 lane 0: {}", tv_warp1_lane0)
    tidx = (None, 0)
    tv_warp0_allLanes = tv_tsr.layout(tidx)
    cute.printf(">?? Value Index for warp 0 all lanes: {}", tv_warp0_allLanes)
    tidx = (0, None)
    tv_lane0 = tv_tsr.layout(tidx)
    cute.printf(">?? Value Index for lane 0 all warps: {}", tv_lane0)

m, n = 512, 512
dtype = cutlass_torch.dtype(cutlass.Int32)
gmem_torch = make_tensor(m, n, dtype)
gmem_tensor = (
    from_dlpack(gmem_torch)
    .mark_layout_dynamic(leading_dim=1)
)
kernel_tiling_examples(gmem_tensor)
