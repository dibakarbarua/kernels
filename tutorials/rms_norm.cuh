/*
------ Row-wise RMS Normalization Kernel for a 2D tensor -------- 
In ML models, most common application of normalization is along the model-dimension.
In LLMs, this model dimension is head_dim * num_heads = d_model

For a large scale LLM this value can be 16K, 32K, 64K... for one layer

Let's say row_size = 32K
- Now we can do row-wise normalization using either vertical or horizontal warps.
- We usually do this in 2-passes
    First Pass: Online Welford Reduction and Welford Combine
    Second Pass: Normalization
- Our primary methods of parallelization are to 
    - process multiple rows per SIMD-engine to hide latency
        - If each SIMD-engine is responsible for computations
    - keep each SIMD-lane's sums in registers
    - Once all lanes are done, reduce across lanes
    - Once all engines are done, reduce across engines using shared memory
For 32K elements, 32 rows per SIMD-engine, and 2B data type, size = 1MB
    This is too large to fit in one SIMD Processor's SMEM
    So we have to do multiple passes through the data

First Pass:
    1. Read each element for a block that fits in SMEM
        2. Each lane reduces its elements
        3. Once each lane is done, each engine reduces across lanes
            4. Engine writes sum to SMEM
    2. Once all elements for a give row are complete for one SIMD engine, synchronize
    // Low utilization phase
    3. One SIMD engine will reduce across engines using SMEM

Synchronize: final_row_sum_sq is available

Second Pass:
    5. Read each element again and normalize

CALCULATIONS FOR WARP/ENGINE PARALLELISM
1. Warps across rows
- Can cover more rows (for more batches and longer sequences)
- Extra latency for swizzle calculations (or extra live registers if hoisted)
LOADS_PER_WARP to hide SMEM latency = 32
    - 32 rows * 32 lanes * 8B = 8KB per warp
4 warps = 32KB per CTA
64 KB is enough to hide GMEM latency for 6B/cyc

Tunables:
numRowsPerCTA
numStagesPerCTA

If we map each of the 32 warp loads to same row, each CTA will process the entire d_model in 
    32K/(numStagesPerCTA * 4 * 2 * 32) cycles
    This is because 4 warps per CTA, 2 elements per lane (packed) 
    ... and 32 lanes per warp.
    = 128/numStagesPerCTA = 64 cycles for 2 stages
*/