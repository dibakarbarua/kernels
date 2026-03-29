/*
----- Batched GEMV Operation to simulate Attention-Decode Kernels ------
Workload:
- We simulate token-by-token decode for LLM(s) using a simulated GEMV
- A = [1,128] B = [seq_len,128], assuming fixed length sequence to avoid workload balancing complexity
- We parallelize across batches and heads, num_work_items = num_batches * num_heads
- Each CTA will work on a full sequence for it's assigned work items
    - We shall assume we have enough work to fully utilize the GPU
- This is a memory bound kernel with an arithmetic intensiy of ~2.
    - We intend to saturate the memory bandwidth of the GPU

Data Movement:
- First we move data from HBM to SMEM and stage A/B tensors for configurable "Stages"
    - This allows us to enforce per-CTA latency hiding
    - Multiple CTAs per SM is another tuning dimension for hiding latency
- A Staging: Multiple sequences per CTA
- B Staging: Multiple tiles per CTA: Latency Hiding

SIMD/SIMT Intrinsics
- We assume each CTA contains 4 warps to fully utilize all Execution Engines
- Each warp works on tile_len/4 individual dot products, no cross-warp dependency
- For one sequence, each warp keeps one register to store sum and then butterfly reduce across lanes
- seq_len is divided into iterations of tile_len tokens each
- NO __syncthreads()! Each warp brings its own data-stage and does not cross compute with another warp.
- A and B are in float16 accuracy and we use FMA f16x2 packed ops per lane
    - To reduce a 128-element hidden dimension each lane has to process 4 elements i.e 2xfp16x2 FMA instructions

Kernel Traits:
- kNumStagesPerWarp is programmed by host
- kTileLen is a kernel trait
- kEmbeddingDim is a kernel trait (reduction-dim)
- kWarpsPerBlock is a kernel trait (4)
- A = 1 x kEmbeddingDim
- B = kTileLen x kEmbeddingDim
- C = 1 x kTileLen
- kBytesPerLane is a kernel trait (4B for NVIDIA SIMT)

Workload Partioning:
- Each Block (CTA) will work on a full sequence
- BatchSize (B) = Number of sequences in workload (num_heads x num_batches in attention terms)
- Sequences per CTA = BatchSize / num_CTAs

Shared Memory:
- A
- B * num_stages

Per-CTA Work:
- Only warp0 will load A (1xEmbeddingDim into shared memory)
- All warps will iterate through B-Tile(s) and load into SMEM
    - We only need to "wait" for one stage though before computing
- All warps will load A tile and B Tile for a stage
    - Compute FMA across 128 elements (2-iterations) using __hfma2
    - Store in one iteration (8B per lane)

PTX Instructions Used:
- cp.async.cg.shared.global
- cp.async.commit_group
- cp.async.wait_group
- fma.rn.f16x2
*/