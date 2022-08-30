#ifndef L3STER_UTIL_COMMON_HPP
#define L3STER_UTIL_COMMON_HPP

#include "l3ster/util/Concepts.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <limits>
#include <memory>
#include <tuple>
#include <type_traits>
#include <vector>

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
    requires(std::is_enum_v< decltype(V) >)
struct EnumTag
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
constexpr bool exactlyOneOf(T... args)
    requires(std::convertible_to< T, bool > and ...)
{
    return (static_cast< size_t >(static_cast< bool >(args)) + ...) == 1u;
}

// Workaround: GCC Bug 97930
// Needed as template parameter because libstdc++ std::pair is not structural (private base class)
// TODO: rely on std::pair once fixed upstream
template < typename T1, typename T2 >
struct Pair
{
    using first_type  = T1;
    using second_type = T2;

    constexpr Pair()
        requires(std::is_default_constructible_v< T1 > and std::is_default_constructible_v< T2 >)
    = default;
    constexpr Pair(T1 t1, T2 t2) : first{std::move(t1)}, second{std::move(t2)} {}

    T1 first;
    T2 second;
};

template < std::integral T >
std::vector< T > concatVectors(std::vector< T > v1, const std::vector< T >& v2)
{
    const auto v1_size_old = v1.size();
    v1.resize(v1.size() + v2.size());
    std::ranges::copy(v2, std::next(begin(v1), v1_size_old));
    return v1;
}

template < typename T >
struct alignas(std::max< std::size_t >(64u /* cacheline size */, alignof(T))) CacheAligned
{
    template < typename... Args >
    constexpr CacheAligned(Args&&... args)
        requires std::constructible_from< T, Args... >
    : value{std::forward< Args >(args)...}
    {}

    constexpr T&       operator*() noexcept { return value; }
    constexpr const T& operator*() const noexcept { return value; }
    constexpr T*       operator->() noexcept { return std::addressof(value); }
    constexpr const T* operator->() const noexcept { return std::addressof(value); }

private:
    T value;
};

template < std::ranges::sized_range auto inds, typename T, size_t N, std::indirectly_writable< T > Iter >
Iter copyValuesAtInds(const std::array< T, N >& array, Iter out_iter)
    requires std::convertible_to< std::ranges::range_value_t< decltype(inds) >, size_t > and
             (std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    for (auto i : inds)
        *out_iter++ = array[i];
    return out_iter;
}

template < std::ranges::sized_range auto inds, typename T, size_t N >
std::array< T, std::ranges::size(inds) > getValuesAtInds(const std::array< T, N >& array)
    requires std::convertible_to< std::ranges::range_value_t< decltype(inds) >, size_t > and
             (std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    std::array< T, std::ranges::size(inds) > retval;
    copyValuesAtInds< inds >(array, begin(retval));
    return retval;
}

enum struct Space
{
    X,
    Y,
    Z
};
} // namespace lstr
#endif // L3STER_UTIL_COMMON_HPP
