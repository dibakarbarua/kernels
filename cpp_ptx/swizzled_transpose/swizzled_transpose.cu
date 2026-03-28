#include <cmath>
#include <cstdlib>
#include <cstdint>
#include <iostream>
#include <limits>
#include <string>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "swizzled_transpose.h"

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
    int rows = 0;
    int cols = 0;
    int iterations = 0;
    std::string element_type;
};

void print_usage(char const* program_name)
{
    std::cerr
        << "Usage: " << program_name
        << " <rows> <cols> <element_type> <iterations>\n"
        << "  element_type: float | int32 | uint32\n";
}

Options parse_options(int argc, char** argv)
{
    if (argc != 5)
    {
        print_usage(argv[0]);
        std::exit(EXIT_FAILURE);
    }

    Options options{};
    options.rows = std::stoi(argv[1]);
    options.cols = std::stoi(argv[2]);
    options.element_type = argv[3];
    options.iterations = std::stoi(argv[4]);

    if (options.rows <= 0 || options.cols <= 0 || options.iterations <= 0)
    {
        std::cerr << "rows, cols, and iterations must all be positive.\n";
        std::exit(EXIT_FAILURE);
    }

    return options;
}

template <typename T>
void reference_transpose(thrust::host_vector<T> const& input,
                         thrust::host_vector<T>& output,
                         int rows,
                         int cols)
{
    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            output[col * rows + row] = input[row * cols + col];
        }
    }
}

template <typename T>
bool compare_results(thrust::host_vector<T> const& actual,
                     thrust::host_vector<T> const& expected,
                     double atol = 1.0e-5)
{
    if (actual.size() != expected.size())
    {
        return false;
    }

    for (size_t idx = 0; idx < actual.size(); ++idx)
    {
        if constexpr (std::is_floating_point_v<T>)
        {
            double const diff =
                std::fabs(static_cast<double>(actual[idx] - expected[idx]));
            if (diff > atol)
            {
                std::cerr << "Mismatch at index " << idx << ": actual="
                          << actual[idx] << " expected=" << expected[idx]
                          << " diff=" << diff << '\n';
                return false;
            }
        }
        else if (actual[idx] != expected[idx])
        {
            std::cerr << "Mismatch at index " << idx << ": actual=" << actual[idx]
                      << " expected=" << expected[idx] << '\n';
            return false;
        }
    }

    return true;
}

template <typename T>
T make_input_value(int row, int col, int cols)
{
    return static_cast<T>(row * cols + col + 1);
}

template <>
float make_input_value<float>(int row, int col, int cols)
{
    return static_cast<float>(row * cols + col) + 0.25f;
}

template <typename T>
void run_case(Options const& options)
{
    constexpr int kStagesPerWarp = 1;

    cudaDeviceProp device_props{};
    check_cuda(cudaGetDeviceProperties(&device_props, 0),
               "cudaGetDeviceProperties");

    int const rows = options.rows;
    int const cols = options.cols;
    int const iterations = options.iterations;

    thrust::host_vector<T> h_input(rows * cols);
    thrust::host_vector<T> h_output(cols * rows, T{0});
    thrust::host_vector<T> h_reference(cols * rows, T{0});

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            h_input[row * cols + col] = make_input_value<T>(row, col, cols);
        }
    }

    reference_transpose(h_input, h_reference, rows, cols);

    thrust::device_vector<T> d_input = h_input;
    thrust::device_vector<T> d_output(cols * rows, T{0});

    dim3 const block_dim(
        swizzled_transpose::TransposeTileTraits<T>::kThreadsPerBlock);
    // 48 SMs in the RTX 3080
    dim3 const grid_dim(64, 48);

    int max_active_blocks_per_sm = 0;
    check_cuda(
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &max_active_blocks_per_sm,
            swizzled_transpose::transpose_kernel<T, kStagesPerWarp>,
            static_cast<int>(block_dim.x),
            0),
        "cudaOccupancyMaxActiveBlocksPerMultiprocessor");

    int const num_sms = device_props.multiProcessorCount;
    int const total_blocks = static_cast<int>(grid_dim.x * grid_dim.y);
    int const target_full_gpu_blocks = num_sms * max_active_blocks_per_sm;
    bool const workload_too_small = total_blocks < target_full_gpu_blocks;

    std::cout << "Device: " << device_props.name << '\n';
    std::cout << "Element type: " << options.element_type
              << " (" << sizeof(T) << " bytes)\n";
    std::cout << "Matrix: " << rows << " x " << cols << '\n';
    std::cout << "Iterations: " << iterations << '\n';
    std::cout << "Grid dim: (" << grid_dim.x << ", " << grid_dim.y << ", "
              << grid_dim.z << ")\n";
    std::cout << "Block dim: (" << block_dim.x << ", " << block_dim.y << ", "
              << block_dim.z << ")\n";
    std::cout << "Stages: " << kStagesPerWarp << std::endl;
    std::cout << "SM count: " << num_sms << '\n';
    std::cout << "Max active blocks per SM for this kernel: "
              << max_active_blocks_per_sm << '\n';
    std::cout << "Total CTAs in workload: " << total_blocks << '\n';
    if (workload_too_small)
    {
        std::cout << "Utilization check: workload is likely too small to fully "
                     "utilize the GPU (need about "
                  << target_full_gpu_blocks << " CTAs for full CTA occupancy).\n";
    }
    else
    {
        std::cout << "Utilization check: workload has enough CTAs to cover all "
                     "SMs at the CTA-occupancy level.\n";
    }

    swizzled_transpose::transpose_kernel<T, kStagesPerWarp>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(d_input.data()),
                                  thrust::raw_pointer_cast(d_output.data()),
                                  rows,
                                  cols);
    check_cuda(cudaGetLastError(), "Warmup kernel launch");
    check_cuda(cudaDeviceSynchronize(), "Warmup kernel execution");

    cudaEvent_t start_event{};
    cudaEvent_t stop_event{};
    check_cuda(cudaEventCreate(&start_event), "cudaEventCreate(start)");
    check_cuda(cudaEventCreate(&stop_event), "cudaEventCreate(stop)");

    check_cuda(cudaEventRecord(start_event), "cudaEventRecord(start)");
    for (int iter = 0; iter < iterations; ++iter)
    {
        swizzled_transpose::transpose_kernel<T, kStagesPerWarp>
            <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(d_input.data()),
                                      thrust::raw_pointer_cast(d_output.data()),
                                      rows,
                                      cols);
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

    if (!compare_results(h_output, h_reference))
    {
        std::cerr << "Transpose validation failed.\n";
        std::exit(EXIT_FAILURE);
    }

    double const bytes_per_iteration =
        2.0 * static_cast<double>(rows) * static_cast<double>(cols) *
        static_cast<double>(sizeof(T));
    double const total_bytes = bytes_per_iteration * static_cast<double>(iterations);
    double const elapsed_s = static_cast<double>(elapsed_ms) / 1.0e3;
    double const bandwidth_gbps =
        (elapsed_s > 0.0) ? (total_bytes / elapsed_s) / 1.0e9 : 0.0;

    std::cout << "Average kernel time: "
              << (elapsed_ms / static_cast<float>(iterations)) << " ms\n";
    std::cout << "Effective memory bandwidth: " << bandwidth_gbps << " GB/s\n";
    std::cout << "Transpose validation passed.\n";
}

} // namespace

int main(int argc, char** argv)
{
    Options const options = parse_options(argc, argv);

    if (options.element_type == "float")
    {
        run_case<float>(options);
        return EXIT_SUCCESS;
    }
    if (options.element_type == "int32" || options.element_type == "int")
    {
        run_case<std::int32_t>(options);
        return EXIT_SUCCESS;
    }
    if (options.element_type == "uint32")
    {
        run_case<std::uint32_t>(options);
        return EXIT_SUCCESS;
    }

    std::cerr << "Unsupported element_type: " << options.element_type << '\n';
    print_usage(argv[0]);
    return EXIT_FAILURE;
}
