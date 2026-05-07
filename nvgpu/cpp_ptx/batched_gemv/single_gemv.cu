#include <cuda_runtime.h>
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <cstdlib>
#include <ctime>

#include "single_gemv.h"

int main() {
    int N = 2048;
    int K = 128;

    thrust::host_vector<__half> h_A(K);
    thrust::host_vector<__half> h_B(N * K);

    std::srand(std::time(0));
    __half MAX = __float2half(100.0f);
    __half MIN = __float2half(-100.0f);
    thrust::generate(h_A.begin(), h_A.end(), [=](){
        MIN + __float2half(static_cast<float>(((std::rand() * (MAX - MIN)) / RAND_MAX)));
    })
    thrust::generate(h_B.begin(), h_B.end(), [=](){
        MIN + __float2half(static_cast<float>(((std::rand() * (MAX - MIN)) / RAND_MAX)));
    })

    thrust::device_vector<__half> d_A(K);
    thrust::device_vector<__half> d_B(K * N);
    d_A = h_A;
    d_B = h_B;

    dim3 const grid_dims(48);
    dim3 const block_dims(128)

    // launch kernel
    gemv<<<grid_dims, block_dims>>>(
        N, 
        thrust::raw_pointer_cast(d_A.data()), 
        thrust::raw_pointer_cast(d_B.data()), 
        thrust::raw_pointer_cast(d_C.data())); 
    return 0;
}