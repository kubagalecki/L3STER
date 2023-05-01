#ifndef L3STER_UTIL_COMMON_HPP
#define L3STER_UTIL_COMMON_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <limits>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace lstr
{
template < auto V >
struct ConstexprValue
{
    using type                  = decltype(V);
    static constexpr auto value = V;
};

template < typename... Types >
struct TypePack
{
    static constexpr auto size = sizeof...(Types);
};

template < auto... Vals >
struct ValuePack
{
    static constexpr auto size = sizeof...(Vals);
};

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
    return (static_cast< std::size_t >(static_cast< bool >(args)) + ...) == std::size_t{1};
}

template < std::integral To, std::integral From >
To exactIntegerCast(From from, std::source_location loc = std::source_location::current())
    requires std::convertible_to< From, To >
{
    constexpr auto max_from = static_cast< std::uintmax_t >(std::numeric_limits< From >::max());
    constexpr auto min_from = static_cast< std::intmax_t >(std::numeric_limits< From >::min());
    constexpr auto max_to   = static_cast< std::uintmax_t >(std::numeric_limits< To >::max());
    constexpr auto min_to   = static_cast< std::intmax_t >(std::numeric_limits< To >::min());

    if constexpr (max_from > max_to)
        util::throwingAssert(
            static_cast< std::uintmax_t >(from) <= max_to,
            "The value being converted is greater then the maximum value representable by the target type",
            loc);

    if constexpr (min_from < min_to)
        util::throwingAssert(
            static_cast< std::intmax_t >(from) >= min_to,
            "The value being converted is less than the minimum value representable by the target type",
            loc);

    return static_cast< To >(from);
}

// If std::unique_ptr<T[]> was a range...
template < typename T >
class ArrayOwner
{
public:
    ArrayOwner() = default;
    explicit ArrayOwner(std::size_t size) : m_data{std::make_unique_for_overwrite< T[] >(size)}, m_size{size} {}

    T*          begin() { return m_data.get(); }
    const T*    begin() const { return m_data.get(); }
    T*          end() { return m_data.get() + m_size; }
    const T*    end() const { return m_data.get() + m_size; }
    T*          data() { return m_data.get(); }
    const T*    data() const { return m_data.get(); }
    T&          operator[](std::size_t i) { return m_data[i]; }
    const T&    operator[](std::size_t i) const { return m_data[i]; }
    T&          front() { return m_data[0]; }
    const T&    front() const { return m_data[0]; }
    T&          back() { return m_data[m_size - 1]; }
    const T&    back() const { return m_data[m_size - 1]; }
    std::size_t size() const { return m_size; }

private:
    std::unique_ptr< T[] > m_data;
    std::size_t            m_size{};
};

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

template < IndexRange_c auto inds, typename T, size_t N, std::indirectly_writable< T > Iter >
Iter copyValuesAtInds(const std::array< T, N >& array, Iter out_iter, ConstexprValue< inds > = {})
    requires(std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    for (auto i : inds)
        *out_iter++ = array[i];
    return out_iter;
}

template < IndexRange_c auto inds, typename T, size_t N >
std::array< T, std::ranges::size(inds) > getValuesAtInds(const std::array< T, N >& array,
                                                         ConstexprValue< inds >    inds_ctwrpr = {})
    requires(std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    std::array< T, std::ranges::size(inds) > retval;
    copyValuesAtInds(array, begin(retval), inds_ctwrpr);
    return retval;
}

template < std::default_initializable T >
T& getThreadLocal()
{
    thread_local T value;
    return value;
}

template < std::integral T >
constexpr T intDivRoundUp(T enumerator, T denominator)
{
    const auto rem  = enumerator % denominator;
    const auto quot = enumerator / denominator;
    return rem == 0 ? quot : quot + 1;
}

enum struct Space
{
    X,
    Y,
    Z
};

enum struct Access
{
    ReadOnly,
    ReadWrite
};
} // namespace lstr
#endif // L3STER_UTIL_COMMON_HPP
