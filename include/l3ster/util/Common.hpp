#ifndef L3STER_UTIL_COMMON_HPP
#define L3STER_UTIL_COMMON_HPP

#include <array>
#include <bitset>
#include <concepts>
#include <limits>
#include <tuple>
#include <type_traits>

namespace lstr
{
template < typename... T >
struct OverloadSet : public T...
{
    using T::operator()...;
};
template < typename... T >
OverloadSet(T&&...) -> OverloadSet< T... >;

template < auto V >
requires(std::is_enum_v< decltype(V) >) struct EnumTag
{};

template < typename T1, typename T2 >
constexpr bool isSameObject(T1& o1, T2& o2)
{
    if constexpr (std::is_same_v< T1, T2 >)
        return std::addressof(o1) == std::addressof(o2);
    else
        return false;
}
template < typename T1, typename T2 >
constexpr bool isSameObject(T1&&, T2&&) = delete;

template < typename... T >
constexpr bool exactlyOneOf(T... args) requires(std::convertible_to< T, bool >and...)
{
    return (static_cast< size_t >(static_cast< bool >(args)) + ...) == 1u;
}

template < size_t n_bits >
constexpr size_t bitsetNUllongs()
{
    constexpr auto ull_bits = sizeof(unsigned long long) * 8u;
    return n_bits / ull_bits + static_cast< bool >(n_bits % ull_bits);
}

template < std::size_t N >
auto serializeBitset(const std::bitset< N >& bits)
{
    using ull                    = unsigned long long;
    constexpr auto ull_bits      = sizeof(ull) * 8u;
    constexpr auto required_ulls = bitsetNUllongs< N >();
    const auto     mask          = std::bitset< N >{std::numeric_limits< ull >::max()}; // 0xffffffffffffffff

    std::array< ull, required_ulls > retval;
    std::generate_n(
        rbegin(retval), retval.size(), [&, i = 0ul]() mutable { return (bits >> i++ * ull_bits & mask).to_ullong(); });
    return retval;
}

template < std::size_t N >
auto deserializeBitset(const std::array< unsigned long long, N >& data)
{
    using ull               = unsigned long long;
    constexpr auto ull_bits = sizeof(ull) * 8u;
    using ret_t             = std::bitset< N * ull_bits >;
    ret_t retval;
    for (auto chunk64 : data)
    {
        retval <<= ull_bits;
        retval |= ret_t{chunk64};
    }
    return retval;
}

template < std::size_t N_out, std::size_t N_in >
auto trimBitset(const std::bitset< N_in >& in) requires(N_out <= N_in)
{
    if constexpr (N_in <= sizeof(unsigned long long) * 8u)
        return std::bitset< N_out >{in.to_ullong()};
    else
    {
        std::bitset< N_out > retval;
        for (std::size_t i = 0u; i < N_out; ++i)
            retval[i] = in[i];
        return retval;
    }
}
} // namespace lstr
#endif // L3STER_UTIL_COMMON_HPP
