#include <cmath>
#include <cstdlib>
#include <iostream>

#include <cuda_runtime.h>

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include "swizzled_transpose.h"

namespace {

template <typename T>
void check_cuda(cudaError_t status, char const* operation)
{
    if (status != cudaSuccess)
    {
        std::cerr << operation << " failed: " << cudaGetErrorString(status)
                  << '\n';
        std::exit(EXIT_FAILURE);
    }
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
                     float atol = 1.0e-5f)
{
    if (actual.size() != expected.size())
    {
        return false;
    }

    for (size_t idx = 0; idx < actual.size(); ++idx)
    {
        float const diff =
            std::fabs(static_cast<float>(actual[idx] - expected[idx]));
        if (diff > atol)
        {
            std::cerr << "Mismatch at index " << idx << ": actual=" << actual[idx]
                      << " expected=" << expected[idx] << " diff=" << diff
                      << '\n';
            return false;
        }
    }

    return true;
}

} // namespace

int main()
{
    using Element = float;
    constexpr int kStagesPerWarp = 1;
    constexpr int kTileExtent =
        swizzled_transpose::TransposeTileTraits<Element>::kTileRows;

    int const rows = 64;
    int const cols = 96;

    thrust::host_vector<Element> h_input(rows * cols);
    thrust::host_vector<Element> h_output(cols * rows, Element{0});
    thrust::host_vector<Element> h_reference(cols * rows, Element{0});

    for (int row = 0; row < rows; ++row)
    {
        for (int col = 0; col < cols; ++col)
        {
            h_input[row * cols + col] =
                static_cast<Element>(row * cols + col + 0.25f);
        }
    }

    reference_transpose(h_input, h_reference, rows, cols);

    thrust::device_vector<Element> d_input = h_input;
    thrust::device_vector<Element> d_output(cols * rows, Element{0});

    dim3 const block_dim(
        swizzled_transpose::TransposeTileTraits<Element>::kThreadsPerBlock);
    dim3 const grid_dim((cols + kTileExtent - 1) / kTileExtent,
                        (rows + kTileExtent - 1) / kTileExtent);

    swizzled_transpose::transpose_kernel<Element, kStagesPerWarp>
        <<<grid_dim, block_dim>>>(thrust::raw_pointer_cast(d_input.data()),
                                  thrust::raw_pointer_cast(d_output.data()),
                                  rows,
                                  cols);

    check_cuda<Element>(cudaGetLastError(), "Kernel launch");
    check_cuda<Element>(cudaDeviceSynchronize(), "Kernel execution");

    thrust::copy(d_output.begin(), d_output.end(), h_output.begin());

    if (!compare_results(h_output, h_reference))
    {
        std::cerr << "Transpose validation failed." << '\n';
        return EXIT_FAILURE;
    }

    std::cout << "Transpose validation passed." << '\n';
    return EXIT_SUCCESS;
}
