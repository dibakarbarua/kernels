#pragma once

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_runtime_api.h>

#include <cstdint>
#include <type_traits>

#include <cute/tensor.hpp>

namespace cpp_ptx {
namespace utils {

template <int B, int M, int S>
struct SharedMemorySwizzle
{
    static_assert(B > 0, "B must be positive.");
    static_assert(M >= 0, "M must be non-negative.");
    static_assert(S > 0, "S must be positive.");
    static_assert(B < 32, "B must fit within a 32-bit offset.");
    static_assert(M + B <= 32, "M + B must fit within a 32-bit offset.");
    static_assert(M + S + B <= 32,
                  "The shifted source bit range must fit within a 32-bit offset.");

    static constexpr uint32_t kBitMask = (uint32_t{1} << B) - 1u;

    CUTE_HOST_DEVICE static constexpr uint32_t apply(uint32_t smem_offset_bytes)
    {
        uint32_t const shifted_bits =
            (smem_offset_bytes >> (M + S)) & kBitMask;
        return smem_offset_bytes ^ (shifted_bits << M);
    }
};

template <int B, int M, int S>
CUTE_HOST_DEVICE static constexpr uint32_t
swizzle_smem_offset(uint32_t smem_offset_bytes)
{
    return SharedMemorySwizzle<B, M, S>::apply(smem_offset_bytes);
}

__device__ __forceinline__ uint32_t cast_smem_ptr_to_uint(void const* ptr)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

template <int Bytes>
__device__ __forceinline__ void cp_async_gmem_to_smem(void* smem_ptr,
                                                      void const* gmem_ptr)
{
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp.async only supports 4B, 8B, and 16B transactions.");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (Bytes == 16)
    uint32_t const smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr), "n"(Bytes));
#else
    using AccessType = cute::array_aligned<uint8_t, Bytes, Bytes>;
    *reinterpret_cast<AccessType*>(smem_ptr) =
        *reinterpret_cast<AccessType const*>(gmem_ptr);
#endif
}

template <int Bytes>
__device__ __forceinline__ void cp_async_gmem_to_smem_zfill(void* smem_ptr,
                                                            void const* gmem_ptr,
                                                            bool pred_guard)
{
    static_assert(Bytes == 4 || Bytes == 8 || Bytes == 16,
                  "cp.async only supports 4B, 8B, and 16B transactions.");

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800) && (Bytes == 16)
    uint32_t const smem_addr = cast_smem_ptr_to_uint(smem_ptr);
    int const valid_bytes = pred_guard ? Bytes : 0;
    asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %3;\n"
                 :
                 : "r"(smem_addr), "l"(gmem_ptr), "n"(Bytes), "r"(valid_bytes));
#else
    using AccessType = cute::array_aligned<uint8_t, Bytes, Bytes>;
    if (pred_guard)
    {
        *reinterpret_cast<AccessType*>(smem_ptr) =
            *reinterpret_cast<AccessType const*>(gmem_ptr);
    }
    else
    {
        *reinterpret_cast<AccessType*>(smem_ptr) = AccessType{};
    }
#endif
}

__device__ __forceinline__ void cp_async_commit_group()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.commit_group;\n" : :);
#endif
}

template <int Groups>
__device__ __forceinline__ void cp_async_wait_group()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_group %0;\n" : : "n"(Groups));
#endif
}

__device__ __forceinline__ void cp_async_wait_all()
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 800)
    asm volatile("cp.async.wait_all;\n" : :);
#endif
}

template <typename T>
__device__ __forceinline__ T load_from_scratch(void const* scratch_ptr,
                                               uint32_t scratch_offset_bytes)
{
    return *reinterpret_cast<T const*>(
        reinterpret_cast<char const*>(scratch_ptr) + scratch_offset_bytes);
}

template <typename T>
__device__ __forceinline__ void store_to_global(T* gmem_ptr,
                                                uint32_t gmem_offset_elements,
                                                T value,
                                                bool pred_guard)
{
    if (pred_guard)
    {
        gmem_ptr[gmem_offset_elements] = value;
    }
}

template <typename T>
__device__ __forceinline__ T load_from_global(T const* gmem_ptr,
                                              uint32_t gmem_offset_elements,
                                              T oob_value,
                                              bool pred_guard)
{
    return pred_guard ? gmem_ptr[gmem_offset_elements] : oob_value;
}

template <typename T>
__device__ __forceinline__ void store_to_smem(void* smem_ptr,
                                              uint32_t smem_offset_bytes,
                                              T value)
{
    *reinterpret_cast<T*>(reinterpret_cast<char*>(smem_ptr) + smem_offset_bytes) =
        value;
}

template <typename T>
__device__ __forceinline__ T fma(T const& a, T const& b, T const& c)
{
    return a * b + c;
}

__device__ __forceinline__ half2 fma_f16x2(half2 a, half2 b, half2 c)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    half2 out;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(*reinterpret_cast<uint32_t*>(&out))
                 : "r"(*reinterpret_cast<uint32_t const*>(&a)),
                   "r"(*reinterpret_cast<uint32_t const*>(&b)),
                   "r"(*reinterpret_cast<uint32_t const*>(&c)));
    return out;
#else
    return __hfma2(a, b, c);
#endif
}

__device__ __forceinline__ half2 add_f16x2(half2 a, half2 b)
{
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 530)
    half2 out;
    asm volatile("add.rn.f16x2 %0, %1, %2;\n"
                 : "=r"(*reinterpret_cast<uint32_t*>(&out))
                 : "r"(*reinterpret_cast<uint32_t const*>(&a)),
                   "r"(*reinterpret_cast<uint32_t const*>(&b)));
    return out;
#else
    return __hadd2(a, b);
#endif
}

__device__ __forceinline__ half2 make_zero_f16x2()
{
    return __float2half2_rn(0.0f);
}

__device__ __forceinline__ float horizontal_add_f16x2(half2 value)
{
    float2 const unpacked = __half22float2(value);
    return unpacked.x + unpacked.y;
}

__device__ __forceinline__ float half2ToFloat(half2 value)
{
    return horizontal_add_f16x2(value);
}

} // namespace utils
} // namespace cpp_ptx
