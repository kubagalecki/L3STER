#ifndef L3STER_UTIL_COMMON_HPP
#define L3STER_UTIL_COMMON_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <concepts>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <numeric>
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

namespace detail
{
template < std::uintmax_t max_representable, std::intmax_t min_representable >
struct SmallestIntegralImpl
{
    using type = decltype(std::invoke([] {
        if constexpr (min_representable < 0)
        {
            if constexpr (max_representable <= std::numeric_limits< std::int8_t >::max() and
                          min_representable >= std::numeric_limits< std::int8_t >::min())
                return std::int8_t{};
            else if constexpr (max_representable <= std::numeric_limits< std::int16_t >::max() and
                               min_representable >= std::numeric_limits< std::int16_t >::min())
                return std::int16_t{};
            else if constexpr (max_representable <= std::numeric_limits< std::int32_t >::max() and
                               min_representable >= std::numeric_limits< std::int32_t >::min())
                return std::int32_t{};
            else if constexpr (max_representable <= std::numeric_limits< std::int64_t >::max() and
                               min_representable >= std::numeric_limits< std::int64_t >::min())
                return std::int64_t{};
        }
        else
        {
            if constexpr (max_representable <= std::numeric_limits< std::uint8_t >::max())
                return std::uint8_t{};
            else if constexpr (max_representable <= std::numeric_limits< std::uint16_t >::max())
                return std::uint16_t{};
            else if constexpr (max_representable <= std::numeric_limits< std::uint32_t >::max())
                return std::uint32_t{};
            else if constexpr (max_representable <= std::numeric_limits< std::uint64_t >::max())
                return std::uint64_t{};
        }
    }));
};
} // namespace detail

template < std::uintmax_t max_representable, std::intmax_t min_representable = 0 >
    requires(not std::same_as< typename detail::SmallestIntegralImpl< max_representable, min_representable >::type,
                               void >)
using smallest_integral_t = detail::SmallestIntegralImpl< max_representable, min_representable >::type;

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

template < std::size_t N >
auto toStdBitset(const std::array< bool, N >& a) -> std::bitset< N >
{
    auto retval = std::bitset< N >{};
    for (auto&& [i, value] : a | std::views::enumerate)
        if (value)
            retval.set(i);
    return retval;
}

template < size_t N, std::floating_point T >
constexpr auto linspaceArray(T lo, T hi) -> std::array< T, N >
    requires(N >= 2)
{
    const auto L      = hi - lo;
    const auto d      = L / static_cast< T >(N - 1);
    auto       retval = std::array< T, N >{};
    std::ranges::transform(
        std::views::iota(0uz, N), retval.begin(), [&](size_t i) { return lo + static_cast< T >(i) * d; });
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
    std::ranges::transform(std::views::iota(0uz, N), std::back_inserter(retval), [&](size_t i) -> T {
        return lo + static_cast< T >(i) * d;
    });
    return retval;
}

template < size_t N, typename T >
constexpr auto makeFilledArray(const T& value) -> std::array< T, N >
{
    std::array< T, N > retval;
    retval.fill(value);
    return retval;
}

template < typename T, size_t N, typename UnaryOp >
constexpr auto elwise(const std::array< T, N >& a, UnaryOp&& op)
    requires std::invocable< UnaryOp, const T& >
{
    using Ret = std::invoke_result_t< UnaryOp, const T& >;
    std::array< Ret, N > retval;
    std::ranges::transform(a, retval.begin(), std::forward< UnaryOp >(op));
    return retval;
}

template < typename T1, typename T2, size_t N, typename BinaryOp >
constexpr auto elwise(const std::array< T1, N >& a, const std::array< T2, N >& b, BinaryOp&& op)
    requires std::invocable< BinaryOp, const T1&, const T2& >
{
    using Ret = std::invoke_result_t< BinaryOp, const T1&, const T2& >;
    std::array< Ret, N > retval;
    std::ranges::transform(a, b, retval.begin(), std::forward< BinaryOp >(op));
    return retval;
}

template < typename T, size_t N, typename BinaryOp >
constexpr auto reduce(const std::array< T, N >& a, BinaryOp&& op, T init = T{})
{
    return std::reduce(a.begin(), a.end(), std::move(init), std::forward< BinaryOp >(op));
}

template < typename UnaryOp1, typename UnaryOp2 >
constexpr auto compose(UnaryOp1&& op1, UnaryOp2&& op2)
{
    return [g = std::forward< UnaryOp1 >(op1), h = std::forward< UnaryOp2 >(op2)](auto&& arg) {
        return std::invoke(g, std::invoke(h, std::forward< decltype(arg) >(arg)));
    };
}

template < typename BinaryOp >
constexpr auto commute(BinaryOp&& op)
{
    return [op = std::forward< BinaryOp >(op)](auto&& arg1, auto&& arg2) {
        return std::invoke(op, std::forward< decltype(arg2) >(arg2), std::forward< decltype(arg1) >(arg1));
    };
}

template < typename BinaryOp >
constexpr auto selfie(BinaryOp&& op)
{
    return [op = std::forward< BinaryOp >(op)](auto&& arg) {
        return std::invoke(op, arg, arg);
    };
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

struct Min
{
    template < typename T >
    T operator()(const T& a, const T& b)
    {
        return b > a ? a : b;
    }
};

struct Max
{
    template < typename T >
    T operator()(const T& a, const T& b)
    {
        return a < b ? b : a;
    }
};

struct AtomicSumInto
{
    template < Arithmetic_c T >
    void operator()(T& value, T update)
    {
        static_assert(std::atomic< T >::is_always_lock_free);
        std::atomic_ref{value}.fetch_add(update, std::memory_order_relaxed);
    }
};

struct AtomicOrInto
{
    template < Arithmetic_c T >
    void operator()(T& value, T update)
    {
        static_assert(std::atomic< T >::is_always_lock_free);
        std::atomic_ref{value}.fetch_or(update, std::memory_order_relaxed);
    }
};
} // namespace lstr::util
#endif // L3STER_UTIL_COMMON_HPP
