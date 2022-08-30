#ifndef L3STER_UTIL_BASE64_HPP
#define L3STER_UTIL_BASE64_HPP

#include "tbb/tbb.h"

#include <array>
#include <concepts>
#include <limits>
#include <memory>
#include <numeric>
#include <ranges>
#include <span>

#if defined(__AVX2__)
#include <immintrin.h>
#endif

namespace lstr
{
namespace detail::b64
{
inline constexpr auto conv_table = [] {
    std::array< char, 64 > table;
    std::iota(begin(table), std::next(begin(table), 26), 'A');
    std::iota(std::next(begin(table), 26), std::next(begin(table), 52), 'a');
    std::iota(std::next(begin(table), 52), std::next(begin(table), 62), '0');
    table[62] = '+';
    table[63] = '/';
    return table;
}();
inline char enc0(std::byte b0)
{
    return conv_table[std::to_integer< unsigned char >(b0 >> 2)];
}
inline char enc1(std::byte b0, std::byte b1)
{
    return conv_table[std::to_integer< unsigned char >(((b0 & std::byte{0x3}) << 4) | (b1 >> 4))];
}
inline char enc2(std::byte b1, std::byte b2)
{
    return conv_table[std::to_integer< unsigned char >(((b1 & std::byte{0xf}) << 2) | (b2 >> 6))];
}
inline char enc3(std::byte b2)
{
    return conv_table[std::to_integer< unsigned char >(b2 & std::byte{0x3f})];
}
inline std::size_t encB64SerialImpl(std::span< const std::byte > data, char*& out)
{
    std::size_t i = 0;
    for (; i + 3 <= data.size(); i += 3)
    {
        const auto b0 = data[i];
        const auto b1 = data[i + 1];
        const auto b2 = data[i + 2];
        *out++        = enc0(b0);
        *out++        = enc1(b0, b1);
        *out++        = enc2(b1, b2);
        *out++        = enc3(b2);
    }
    return i;
}
inline void encB64Remainder(std::span< const std::byte > data, char*& out)
{
    if (data.size() == 1)
    {
        const auto last_byte = data.back();
        *out++               = enc0(last_byte);
        *out++               = enc1(last_byte, std::byte{0});
        *out++               = '=';
        *out++               = '=';
    }
    else if (data.size() == 2)
    {
        const auto penult_byte = data.front();
        const auto last_byte   = data.back();
        *out++                 = enc0(penult_byte);
        *out++                 = enc1(penult_byte, last_byte);
        *out++                 = enc2(last_byte, std::byte{0});
        *out++                 = '=';
    }
}
inline std::size_t encB64SimdImpl(std::span< const std::byte > data, char*& out)
{
#if defined(__AVX2__)
    // Based on: https://doi.org/10.1145/3132709
    constexpr auto load24 = [](const std::byte* begin) {
        constexpr auto fal       = 0;
        constexpr auto tru       = -1;
        const auto     load_mask = _mm256_setr_epi32(fal, tru, tru, tru, tru, tru, tru, fal);
        const auto     load_addr = reinterpret_cast< const int* >(std::prev(begin, sizeof(int)));
        return _mm256_maskload_epi32(load_addr, load_mask);
    };
    constexpr auto unpack_lo6 = [](__m256i packed) {
        const auto shuffle_mask = _mm256_set_epi8(
            10, 11, 9, 10, 7, 8, 6, 7, 4, 5, 3, 4, 1, 2, 0, 1, 14, 15, 13, 14, 11, 12, 10, 11, 8, 9, 7, 8, 5, 6, 4, 5);
        const auto c0 = _mm256_set1_epi32(0x0fc0fc00);
        const auto c1 = _mm256_set1_epi32(0x04000040);
        const auto c2 = _mm256_set1_epi32(0x003f03f0);
        const auto c3 = _mm256_set1_epi32(0x01000010);

        const auto shuf = _mm256_shuffle_epi8(packed, shuffle_mask);
        const auto t0   = _mm256_and_si256(shuf, c0);
        const auto t1   = _mm256_mulhi_epu16(t0, c1);
        const auto t2   = _mm256_and_si256(shuf, c2);
        const auto t3   = _mm256_mullo_epi16(t2, c3);
        return _mm256_or_si256(t1, t3);
    };
    constexpr auto to_ascii = [](__m256i b64_inds) {
        const auto v13        = _mm256_set1_epi8(13);
        const auto v26        = _mm256_set1_epi8(26);
        const auto v51        = _mm256_set1_epi8(51);
        const auto offset_map = _mm256_setr_epi8(71,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -19,
                                                 -16,
                                                 65,
                                                 0,
                                                 0,
                                                 71,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -4,
                                                 -19,
                                                 -16,
                                                 65,
                                                 0,
                                                 0);

        const auto lt26_mask = _mm256_cmpgt_epi8(v26, b64_inds);
        const auto red_0_25  = _mm256_and_si256(lt26_mask, v13);
        const auto red_26_63 = _mm256_subs_epu8(b64_inds, v51);
        const auto red_0_63  = _mm256_or_si256(red_26_63, red_0_25);
        const auto offsets   = _mm256_shuffle_epi8(offset_map, red_0_63);
        return _mm256_add_epi8(b64_inds, offsets);
    };

    constexpr std::size_t block_size       = 24;
    constexpr std::size_t block_proc_bytes = 32;
    const auto            n_blocks         = data.size() / block_size;
    const auto            block_range      = tbb::blocked_range< std::size_t >{0, n_blocks};
    const auto            process_block    = [&](std::size_t block_num) {
        const auto block_start    = std::next(data.data(), block_num * block_size);
        const auto out_start      = std::next(out, block_num * block_proc_bytes);
        const auto packed_block   = load24(block_start);
        const auto unpacked_block = unpack_lo6(packed_block);
        const auto ascii_block    = to_ascii(unpacked_block);
        _mm256_storeu_si256(reinterpret_cast< __m256i* >(out_start), ascii_block);
    };
    tbb::parallel_for(block_range, [&](const tbb::blocked_range< std::size_t >& range) {
        for (auto block = range.begin(); block != range.end(); ++block)
            process_block(block);
    });
    std::advance(out, n_blocks * block_proc_bytes);
    return n_blocks * block_size;
#else
    return 0;
#endif
}
inline std::size_t alignForSimd(std::span< const std::byte > data, char*& out)
{
    static_assert(alignof(int) == 4);
    auto ptr   = (void*)data.data();
    auto space = std::numeric_limits< std::size_t >::max();
    std::align(alignof(int), 1, ptr, space);
    auto misaligned_by = std::numeric_limits< std::size_t >::max() - space;
    misaligned_by += (misaligned_by % 3) * alignof(int); // Alignment pass can't leave remainder
    if (misaligned_by <= data.size())
        return encB64SerialImpl(data.subspan(0, misaligned_by), out);
    else
        return 0;
}
} // namespace detail::b64
template < std::ranges::contiguous_range R, std::contiguous_iterator I >
std::size_t encodeAsBase64(R&& data, I out_it)
    requires std::ranges::sized_range< R > and std::same_as< std::iter_value_t< I >, char >
{
    const auto  data_span       = std::span{std::ranges::data(data), std::ranges::size(data)};
    const auto  byte_span       = std::as_bytes(data_span);
    auto        out_ptr         = std::addressof(*out_it);
    std::size_t bytes_processed = 0;

#if defined(__AVX2__)
    // Note: it is unclear whether `vpmaskmov` supports unaligned access, better to err on the side of caution
    if constexpr (alignof(std::ranges::range_value_t< R >) < alignof(int))
        bytes_processed += detail::b64::alignForSimd(byte_span, out_ptr);
#endif

    bytes_processed += detail::b64::encB64SimdImpl(byte_span.subspan(bytes_processed), out_ptr);
    bytes_processed += detail::b64::encB64SerialImpl(byte_span.subspan(bytes_processed), out_ptr);
    detail::b64::encB64Remainder(byte_span.subspan(bytes_processed), out_ptr);
    const auto bytes_written = std::distance(std::addressof(*out_it), out_ptr);
    return bytes_written;
}

template < typename T >
std::size_t getBase64EncodingSize(std::size_t size)
{
    const auto bytes           = size * sizeof(T);
    const auto n_full_triplets = bytes / 3;
    const auto remainder       = bytes % 3;
    return (n_full_triplets + (remainder > 0)) * 4;
}
} // namespace lstr
#endif // L3STER_UTIL_BASE64_HPP
