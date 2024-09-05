#ifndef L3STER_UTIL_SIMD_HPP
#define L3STER_UTIL_SIMD_HPP

#include <cstdint>

#if __has_include(<immintrin.h>)
#include <immintrin.h>
#endif

namespace lstr::util
{
#if defined(__AVX512__)
inline constexpr std::size_t simd_width = 64;
#else
#if defined(__AVX__)
inline constexpr std::size_t simd_width = 32;
#else
#if defined(__SSE__)
inline constexpr std::size_t simd_width = 16;
#else
inline constexpr std::size_t simd_width = 8;
#endif
#endif
#endif
} // namespace lstr::util
#endif // L3STER_UTIL_SIMD_HPP
