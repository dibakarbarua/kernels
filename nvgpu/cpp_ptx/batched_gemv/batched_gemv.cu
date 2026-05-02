#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <string>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "batched_gemv.h"

namespace {

void check_cuda(cudaError_t status, char const* operation)
{
    if (status != cudaSuccess)
    {
        std::cerr << operation << " failed: " << cudaGetErrorString(status)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
}

struct Options
{
    int num_batches = 0;
    int seq_len = 0;
    int iterations = 0;
    int stages = 1;
};

void print_usage(char const* program_name)
{
    std::cerr << "Usage: " << program_name
              << " <num_batches> <seq_len> <iterations> <stages>\n";
    std::cerr << "  stages: currently only 1 is supported by the static shared-memory layout.\n";
}

Options parse_options(int argc, char** argv)
{
    if (argc != 5)
    {
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    Options options{};
    options.num_batches = std::stoi(argv[1]);
    options.seq_len = std::stoi(argv[2]);
    options.iterations = std::stoi(argv[3]);
    options.stages = std::stoi(argv[4]);

    if (options.num_batches <= 0 || options.seq_len <= 0 ||
        options.iterations <= 0 || options.stages <= 0)
    {
        std::cerr
            << "num_batches, seq_len, iterations, and stages must all be positive.\n";
        std::exit(EXIT_FAILURE);
    }

    return options;
}

template <typename T>
T make_query_value(int batch_idx, int elem_idx)
{
    return __float2half(
        0.01f * static_cast<float>((batch_idx + 1) * ((elem_idx % 17) + 1)));
}

template <typename T>
T make_key_value(int batch_idx, int seq_idx, int elem_idx)
{
    return __float2half(0.001f * static_cast<float>(
                                   ((batch_idx % 5) + 1) * ((seq_idx % 13) + 1) *
                                   ((elem_idx % 19) + 1)));
}

void reference_batched_gemv(thrust::host_vector<half> const& query,
                            thrust::host_vector<half> const& key,
                            thrust::host_vector<float>& output,
                            int num_batches,
                            int seq_len,
                            int embedding_dim)
{
    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx)
        {
            float acc = 0.0f;
            for (int elem_idx = 0; elem_idx < embedding_dim; ++elem_idx)
            {
                size_t const query_offset =
                    static_cast<size_t>(batch_idx) * embedding_dim + elem_idx;
                size_t const key_offset =
                    (static_cast<size_t>(batch_idx) * seq_len + seq_idx) *
                        embedding_dim +
                    elem_idx;
                acc += __half2float(query[query_offset]) *
                       __half2float(key[key_offset]);
            }
            output[static_cast<size_t>(batch_idx) * seq_len + seq_idx] = acc;
        }
    }
}

bool compare_results(thrust::host_vector<float> const& actual,
                     thrust::host_vector<float> const& expected,
                     double atol = 2.0e-2)
{
    if (actual.size() != expected.size())
    {
        return false;
    }

    for (size_t idx = 0; idx < actual.size(); ++idx)
    {
        double const diff =
            std::fabs(static_cast<double>(actual[idx] - expected[idx]));
        if (diff > atol)
        {
            std::cerr << "Mismatch at index " << idx << ": actual=" << actual[idx]
                      << " expected=" << expected[idx] << " diff=" << diff << '\n';
            return false;
        }
    }

    return true;
}

template <int Stages>
void run_case(Options const& options)
{
    using Element = half;
    using Traits = batched_gemv::GemvKernelTraits<Element>;

    cudaDeviceProp device_props{};
    check_cuda(cudaGetDeviceProperties(&device_props, 0),
               "cudaGetDeviceProperties");

    int const num_batches = options.num_batches;
    int const seq_len = options.seq_len;
    int const iterations = options.iterations;
    int const embedding_dim = Traits::kEmbeddingDim;

    thrust::host_vector<Element> h_query(num_batches * embedding_dim);
    thrust::host_vector<Element> h_key(num_batches * seq_len * embedding_dim);
    thrust::host_vector<float> h_output(num_batches * seq_len, 0.0f);
    thrust::host_vector<float> h_reference(num_batches * seq_len, 0.0f);

    for (int batch_idx = 0; batch_idx < num_batches; ++batch_idx)
    {
        for (int elem_idx = 0; elem_idx < embedding_dim; ++elem_idx)
        {
            h_query[static_cast<size_t>(batch_idx) * embedding_dim + elem_idx] =
                make_query_value<Element>(batch_idx, elem_idx);
        }

        for (int seq_idx = 0; seq_idx < seq_len; ++seq_idx)
        {
            for (int elem_idx = 0; elem_idx < embedding_dim; ++elem_idx)
            {
                h_key[(static_cast<size_t>(batch_idx) * seq_len + seq_idx) *
                          embedding_dim +
                      elem_idx] =
                    make_key_value<Element>(batch_idx, seq_idx, elem_idx);
            }
        }
    }

    reference_batched_gemv(
        h_query, h_key, h_reference, num_batches, seq_len, embedding_dim);

    thrust::device_vector<Element> d_query = h_query;
    thrust::device_vector<Element> d_key = h_key;
    thrust::device_vector<float> d_output(num_batches * seq_len, 0.0f);

    dim3 const block_dim(Traits::kThreadsPerBlock);

    int max_active_blocks_per_sm = 0;
    check_cuda(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                   &max_active_blocks_per_sm,
                   batched_gemv::batched_gemv_kernel<Stages, Element, Traits>,
                   static_cast<int>(block_dim.x),
                   0),
               "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    int const num_sms = device_props.multiProcessorCount;
    int const target_grid_x = num_sms * max_active_blocks_per_sm;
    int const grid_x = 48;
    dim3 const grid_dim(grid_x);

    bool const workload_too_small = num_batches < target_grid_x;

    std::cout << "Device: " << device_props.name << '\n';
    std::cout << "Input type: half (" << sizeof(Element) << " bytes)\n";
    std::cout << "Batches: " << num_batches << '\n';
    std::cout << "Sequence length: " << seq_len << '\n';
    std::cout << "Embedding dim: " << embedding_dim << '\n';
    std::cout << "Tile len: " << Traits::kTileLen << '\n';
    std::cout << "Stages: " << Stages << '\n';
    std::cout << "Iterations: " << iterations << '\n';
    std::cout << "Grid dim: (" << grid_dim.x << ", " << grid_dim.y << ", "
              << grid_dim.z << ")\n";
    std::cout << "Block dim: (" << block_dim.x << ", " << block_dim.y << ", "
              << block_dim.z << ")\n";
    std::cout << "SM count: " << num_sms << '\n';
    std::cout << "Max active blocks per SM for this kernel: "
              << max_active_blocks_per_sm << '\n';
    std::cout << "Total CTAs in workload: " << grid_dim.x << '\n';
    if (workload_too_small)
    {
        std::cout << "Utilization check: workload is likely too small to fully "
                     "utilize the GPU (need about "
                  << target_grid_x << " CTAs for full CTA occupancy).\n";
    }
    else
    {
        std::cout << "Utilization check: workload has enough CTAs to cover all "
                     "SMs at the CTA-occupancy level.\n";
    }

    batched_gemv::batched_gemv_kernel<Stages, Element, Traits>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(d_query.data()),
                                  thrust::raw_pointer_cast(d_key.data()),
                                  thrust::raw_pointer_cast(d_output.data()),
                                  num_batches,
                                  seq_len);
    check_cuda(cudaGetLastError(), "Warmup kernel launch");
    check_cuda(cudaDeviceSynchronize(), "Warmup kernel execution");

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");
    for (int iter = 0; iter < iterations; ++iter)
    {
        batched_gemv::batched_gemv_kernel<Stages, Element, Traits>
            <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(d_query.data()),
                                      thrust::raw_pointer_cast(d_key.data()),
                                      thrust::raw_pointer_cast(d_output.data()),
                                      num_batches,
                                      seq_len);
    }
    check_cuda(cudaEventRecord(stop_event), "cudaEventRecord(stop)");
    check_cuda(cudaGetLastError(), "Timed kernel launch");
    check_cuda(cudaEventSynchronize(stop_event), "cudaEventSynchronize(stop)");

    float elapsed_ms = 0.0f;
    check_cuda(cudaEventElapsedTime(&elapsed_ms, start_event, stop_event),
               "cudaEventElapsedTime");
    check_cuda(cudaEventDestroy(start_event), "cudaEventDestroy(start)");
    check_cuda(cudaEventDestroy(stop_event), "cudaEventDestroy(stop)");

    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    if (!compare_results(h_output, h_reference, 0.1))
    {
        std::cerr << "Batched GEMV validation failed.\n";
        std::exit(EXIT_FAILURE);
    }

    double const query_bytes =
        static_cast<double>(num_batches) * embedding_dim * sizeof(Element);
    double const key_bytes = static_cast<double>(num_batches) * seq_len *
                             embedding_dim * sizeof(Element);
    double const output_bytes =
        static_cast<double>(num_batches) * seq_len * sizeof(float);
    double const bytes_per_iteration = query_bytes + key_bytes + output_bytes;
    double const total_bytes = bytes_per_iteration * static_cast<double>(iterations);
    double const elapsed_s = static_cast<double>(elapsed_ms) / 1.0e3;
    double const bandwidth_gbps =
        (elapsed_s > 0.0) ? (total_bytes / elapsed_s) / 1.0e9 : 0.0;

    std::cout << "Average kernel time: "
              << (elapsed_ms / static_cast<float>(iterations)) << " ms\n";
    std::cout << "Approx. effective bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << "Batched GEMV validation passed.\n";
}

} // namespace

int main(int argc, char** argv)
{
    Options const options = parse_options(argc, argv);
    run_case<1>(options);
    return EXIT_SUCCESS;
}
