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

// Workaround: GCC Bug 97930
// Needed as template parameter because libstdc++ std::pair is not structural (private base class)
// TODO: rely on std::pair once fixed upstream
template < typename T1, typename T2 >
struct Pair
{
    using first_type  = T1;
    using second_type = T2;

    constexpr Pair() requires(std::is_default_constructible_v< T1 >and std::is_default_constructible_v< T2 >) = default;
    constexpr Pair(T1 t1, T2 t2) : first{std::move(t1)}, second{std::move(t2)} {}

    T1 first;
    T2 second;
};
} // namespace lstr
#endif // L3STER_UTIL_COMMON_HPP
