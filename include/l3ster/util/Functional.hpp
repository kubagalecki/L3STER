#ifndef L3STER_UTIL_FUNCTIONAL_HPP
#define L3STER_UTIL_FUNCTIONAL_HPP

#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/Concepts.hpp"

namespace lstr::util
{
template < typename T, typename F >
auto elwise(ArrayOwner< T > array, F&& fun) -> ArrayOwner< T >
    requires Mapping_c< F, T, T >
{
    for (T& t : array)
        t = fun(std::move(t));
    return array;
}

template < typename T, typename R >
auto reduce(ArrayOwner< T > array, R&& reduction, T init = T{})
{
    return std::reduce(array.begin(), array.end(), std::move(init), std::forward< R >(reduction));
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
#endif // L3STER_UTIL_FUNCTIONAL_HPP
