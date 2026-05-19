#pragma once

#include <cuda_runtime.h>

// Reduces N fp32 values into output[0].
//
// The kernel is intentionally tutorial-sized:
//   1. each thread accumulates a strided partial sum,
//   2. each warp reduces with PTX shfl.sync,
//   3. each CTA reduces those warp sums in normal CTA shared memory,
//   4. each CTA writes one value through a PTX cluster-shared address,
//   5. CTA rank 0 in each cluster reads every peer CTA's value through
//      PTX cluster-shared loads,
//   6. each cluster contributes to output[0] with PTX red.global.add.f32.
//
// Cluster shared memory is a Hopper / SM90 feature. Build this sample with:
//   bash ../build_and_run.sh cluster_shared_memory --sm 90
cudaError_t launch_cluster_shared_memory_reduction(const float* input,
                                                   float* output,
                                                   int n,
                                                   cudaStream_t stream = 0);
