#ifndef L3STER_UTIL_COMMON_HPP
#define L3STER_UTIL_COMMON_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <span>
#include <tuple>
#include <type_traits>
#include <vector>

namespace lstr::util
{
template < auto V >
struct ConstexprValue
{
    using type                  = decltype(V);
    static constexpr auto value = V;
};

#define L3STER_WRAP_CTVAL(val__)                                                                                       \
    ::lstr::util::ConstexprValue< val__ >                                                                              \
    {}

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
OverloadSet(T&&...) -> OverloadSet< std::decay_t< T >... >;

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

template < typename T >
class CachelineAligned
{
public:
    template < typename... Args >
    explicit CachelineAligned(Args&&... args)
        requires std::constructible_from< T, decltype(std::forward< Args >(args))... >
        : m_value(std::forward< Args >(args)...)
    {}

    T&       operator*() noexcept { return m_value; }
    const T& operator*() const noexcept { return m_value; }
    T*       operator->() noexcept { return std::addressof(m_value); }
    const T* operator->() const noexcept { return std::addressof(m_value); }

private:
    static constexpr size_t cacheline_size = 64;
    static constexpr size_t alignment      = std::max(alignof(T), cacheline_size);

    alignas(alignment) T m_value;
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
        requires std::default_initializable< T1 > and std::default_initializable< T2 >
    = default;
    constexpr Pair(T1 t1, T2 t2) : first{std::move(t1)}, second{std::move(t2)} {}

    T1 first;
    T2 second;
};

template < std::default_initializable T >
T& getThreadLocal()
{
    thread_local T value;
    return value;
}

template < size_t N, std::floating_point T >
constexpr auto linspaceArray(T lo, T hi) -> std::array< T, N >
    requires(N >= 2)
{
    const auto L      = hi - lo;
    const auto d      = L / static_cast< T >(N - 1);
    auto       retval = std::array< T, N >{};
    std::ranges::transform(
        std::views::iota(size_t{0}, N), retval.begin(), [&](size_t i) { return lo + static_cast< T >(i) * d; });
    return retval;
}

template < std::floating_point T >
auto linspaceVector(T lo, T hi, size_t N) -> std::vector< T >
{
    util::throwingAssert(N >= 2);
    const auto L      = hi - lo;
    const auto d      = L / static_cast< T >(N - 1);
    auto       retval = std::vector< T >{};
    retval.reserve(N);
    std::ranges::transform(std::views::iota(size_t{0}, N), std::back_inserter(retval), [&](size_t i) -> T {
        return lo + static_cast< T >(i) * d;
    });
    return retval;
}

template < typename Predicate >
constexpr auto negatePredicate(Predicate&& predicate)
{
    return [predicate = std::forward< Predicate >(predicate)]< typename T >(T&& in)
        requires std::predicate< std::add_const_t< std::remove_cvref_t< Predicate > >, decltype(std::forward< T >(in)) >
    {
        return not std::invoke(predicate, std::forward< T >(in));
    };
}

struct AtomicSumInto
{
    template < Arithmetic_c T >
    void operator()(T& value, T update)
    {
        static_assert(std::atomic< T >::is_always_lock_free);
        std::atomic_ref{value}.fetch_add(update, std::memory_order_relaxed);
    }
};
} // namespace lstr::util
#endif // L3STER_UTIL_COMMON_HPP
