#include "cluster_shared_memory.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>

namespace {

constexpr int kThreadsPerBlock = 128;
constexpr int kWarpsPerBlock = kThreadsPerBlock / 32;
constexpr int kCtasPerCluster = 4;

#define CHECK_CUDA(expr)                                                       \
    do {                                                                       \
        cudaError_t status = (expr);                                           \
        if (status != cudaSuccess) {                                           \
            std::fprintf(stderr, "%s:%d: CUDA error: %s\n", __FILE__,          \
                         __LINE__, cudaGetErrorString(status));                \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

__device__ __forceinline__ float ptx_add_f32(float a, float b) {
    float out;
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(out) : "f"(a), "f"(b));
    return out;
}

__device__ __forceinline__ float ptx_ld_global_f32(const float* ptr) {
    float out;
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(out) : "l"(ptr));
    return out;
}

__device__ __forceinline__ void ptx_red_global_add_f32(float* ptr, float value) {
    asm volatile("red.global.add.f32 [%0], %1;" :: "l"(ptr), "f"(value)
                 : "memory");
}

__device__ __forceinline__ void ptx_bar_sync() {
    // CTA-local barrier, equivalent in scope to __syncthreads().
    asm volatile("bar.sync 0;" ::: "memory");
}

__device__ __forceinline__ void ptx_barrier_cluster() {
    // A cluster barrier makes writes from one CTA visible to peer CTAs in the
    // same thread-block cluster before any peer performs cluster-shared loads.
    asm volatile("barrier.cluster.arrive.aligned;" ::: "memory");
    asm volatile("barrier.cluster.wait.aligned;" ::: "memory");
}

__device__ __forceinline__ int ptx_cluster_rank() {
    int rank;
    asm volatile("mov.u32 %0, %%cluster_ctarank;" : "=r"(rank));
    return rank;
}

__device__ __forceinline__ int ptx_cluster_size() {
    int size;
    asm volatile("mov.u32 %0, %%cluster_nctarank;" : "=r"(size));
    return size;
}

__device__ __forceinline__ unsigned ptx_shared_u32addr(const void* ptr) {
    // CUDA C++ gives us a generic pointer. PTX shared-memory instructions want
    // a shared-space address, so cvta.to.shared converts the pointer first.
    unsigned addr;
    asm volatile("{\n"
                 "  .reg .u64 shared_addr;\n"
                 "  cvta.to.shared.u64 shared_addr, %1;\n"
                 "  cvt.u32.u64 %0, shared_addr;\n"
                 "}\n"
                 : "=r"(addr)
                 : "l"(ptr));
    return addr;
}

__device__ __forceinline__ unsigned
ptx_mapa_shared_cluster(unsigned cta_shared_addr, int cta_rank) {
    // mapa.shared::cluster maps the same CTA-local shared-memory offset into
    // the address window of the CTA with rank cta_rank inside this cluster.
    unsigned cluster_addr;
    asm volatile("mapa.shared::cluster.u32 %0, %1, %2;"
                 : "=r"(cluster_addr)
                 : "r"(cta_shared_addr), "r"(cta_rank));
    return cluster_addr;
}

__device__ __forceinline__ void ptx_st_shared_cta_f32(unsigned addr,
                                                      float value) {
    asm volatile("st.shared::cta.f32 [%0], %1;" :: "r"(addr), "f"(value)
                 : "memory");
}

__device__ __forceinline__ float ptx_ld_shared_cta_f32(unsigned addr) {
    float value;
    asm volatile("ld.shared::cta.f32 %0, [%1];"
                 : "=f"(value)
                 : "r"(addr)
                 : "memory");
    return value;
}

__device__ __forceinline__ void ptx_st_shared_cluster_f32(unsigned addr,
                                                          float value) {
    asm volatile("st.shared::cluster.f32 [%0], %1;" :: "r"(addr), "f"(value)
                 : "memory");
}

__device__ __forceinline__ float ptx_ld_shared_cluster_f32(unsigned addr) {
    float value;
    asm volatile("ld.shared::cluster.f32 %0, [%1];"
                 : "=f"(value)
                 : "r"(addr)
                 : "memory");
    return value;
}

__device__ __forceinline__ float ptx_shfl_down_f32(float value, int offset) {
    float shifted;
    asm volatile("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
                 : "=f"(shifted)
                 : "f"(value), "r"(offset));
    return shifted;
}

__device__ __forceinline__ float ptx_warp_reduce_sum(float value) {
    // Only lane 0 is used after this reduction. The intermediate values in
    // higher lanes are not meaningful once their source lane falls outside the
    // warp, which is fine for a lane-0 reduction.
    value = ptx_add_f32(value, ptx_shfl_down_f32(value, 16));
    value = ptx_add_f32(value, ptx_shfl_down_f32(value, 8));
    value = ptx_add_f32(value, ptx_shfl_down_f32(value, 4));
    value = ptx_add_f32(value, ptx_shfl_down_f32(value, 2));
    value = ptx_add_f32(value, ptx_shfl_down_f32(value, 1));
    return value;
}

__global__ void __cluster_dims__(4, 1, 1)
cluster_shared_memory_reduction_kernel(const float* input, float* output,
                                       int n) {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < 900)
#error "cluster_shared_memory requires SM90 or newer. Build with --sm 90."
#endif

    // These are ordinary CTA shared-memory declarations. There is no separate
    // CUDA C++ spelling for "cluster shared memory"; PTX maps a CTA shared
    // address into another CTA's shared-memory window with mapa.shared::cluster.
    __shared__ float warp_sums[kWarpsPerBlock];
    __shared__ float cluster_slot;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;

    float thread_sum = 0.0f;
    for (int i = blockIdx.x * blockDim.x + tid; i < n;
         i += gridDim.x * blockDim.x) {
        thread_sum = ptx_add_f32(thread_sum, ptx_ld_global_f32(input + i));
    }

    const float warp_sum = ptx_warp_reduce_sum(thread_sum);

    if (lane == 0) {
        const unsigned warp_sum_addr = ptx_shared_u32addr(warp_sums + warp);
        ptx_st_shared_cta_f32(warp_sum_addr, warp_sum);
    }

    ptx_bar_sync();

    if (tid == 0) {
        float cta_sum = 0.0f;
        for (int i = 0; i < kWarpsPerBlock; ++i) {
            const unsigned warp_sum_addr = ptx_shared_u32addr(warp_sums + i);
            cta_sum = ptx_add_f32(cta_sum,
                                  ptx_ld_shared_cta_f32(warp_sum_addr));
        }

        // Write this CTA's result through the cluster-shared address window.
        // Mapping to our own rank is intentionally redundant, but it lets this
        // tutorial show the exact write form: mapa + st.shared::cluster.
        const unsigned local_slot = ptx_shared_u32addr(&cluster_slot);
        const unsigned my_cluster_slot =
            ptx_mapa_shared_cluster(local_slot, ptx_cluster_rank());
        ptx_st_shared_cluster_f32(my_cluster_slot, cta_sum);
    }

    ptx_barrier_cluster();

    if (tid == 0 && ptx_cluster_rank() == 0) {
        float cluster_sum = 0.0f;
        const unsigned local_slot = ptx_shared_u32addr(&cluster_slot);

        // CTA rank 0 walks the CTAs in its cluster. For each rank, the same
        // local shared-memory offset is mapped into that peer CTA's shared
        // memory, then read with ld.shared::cluster.
        for (int rank = 0; rank < ptx_cluster_size(); ++rank) {
            const unsigned peer_slot =
                ptx_mapa_shared_cluster(local_slot, rank);
            cluster_sum = ptx_add_f32(
                cluster_sum, ptx_ld_shared_cluster_f32(peer_slot));
        }

        // Each cluster contributes one fp32 add. Across all clusters this gives
        // a complete grid reduction into output[0]. The host zeros output first.
        ptx_red_global_add_f32(output, cluster_sum);
    }
}

int round_up_to_multiple(int value, int multiple) {
    return ((value + multiple - 1) / multiple) * multiple;
}

} // namespace

cudaError_t launch_cluster_shared_memory_reduction(const float* input,
                                                   float* output, int n,
                                                   cudaStream_t stream) {
    if (n < 0) {
        return cudaErrorInvalidValue;
    }

    cudaError_t status = cudaMemsetAsync(output, 0, sizeof(float), stream);
    if (status != cudaSuccess) {
        return status;
    }

    int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
    blocks = std::max(kCtasPerCluster,
                      round_up_to_multiple(blocks, kCtasPerCluster));

    cluster_shared_memory_reduction_kernel<<<blocks, kThreadsPerBlock, 0,
                                             stream>>>(input, output, n);
    return cudaGetLastError();
}

int main(int argc, char** argv) {
    int n = 1 << 20;
    if (argc > 1) {
        n = std::atoi(argv[1]);
    }
    if (n < 0) {
        std::fprintf(stderr, "N must be non-negative.\n");
        return EXIT_FAILURE;
    }

    int device = 0;
    cudaDeviceProp props{};
    CHECK_CUDA(cudaGetDevice(&device));
    CHECK_CUDA(cudaGetDeviceProperties(&props, device));
    if (props.major < 9) {
        std::fprintf(stderr,
                     "This tutorial uses cluster shared memory and needs "
                     "SM90+. Current device is sm_%d%d.\n",
                     props.major, props.minor);
        return EXIT_SUCCESS;
    }

    std::vector<float> host_input(n);
    for (int i = 0; i < n; ++i) {
        host_input[i] = 1.0f + static_cast<float>(i % 7) * 0.25f;
    }

    const float expected =
        std::accumulate(host_input.begin(), host_input.end(), 0.0f);

    float* device_input = nullptr;
    float* device_output = nullptr;
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_input),
                          n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&device_output),
                          sizeof(float)));
    CHECK_CUDA(cudaMemcpy(device_input, host_input.data(), n * sizeof(float),
                          cudaMemcpyHostToDevice));

    CHECK_CUDA(launch_cluster_shared_memory_reduction(device_input,
                                                      device_output, n));
    CHECK_CUDA(cudaDeviceSynchronize());

    float actual = 0.0f;
    CHECK_CUDA(cudaMemcpy(&actual, device_output, sizeof(float),
                          cudaMemcpyDeviceToHost));

    const float abs_error = std::fabs(actual - expected);
    const float tolerance = std::max(1e-2f, std::fabs(expected) * 1e-5f);
    std::printf("cluster shared-memory reduction\n");
    std::printf("  N        : %d\n", n);
    std::printf("  expected : %.6f\n", expected);
    std::printf("  actual   : %.6f\n", actual);
    std::printf("  abs err  : %.6f\n", abs_error);
    std::printf("  tolerance: %.6f\n", tolerance);

    CHECK_CUDA(cudaFree(device_input));
    CHECK_CUDA(cudaFree(device_output));

    return abs_error <= tolerance ? EXIT_SUCCESS : EXIT_FAILURE;
}
