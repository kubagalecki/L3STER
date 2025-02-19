// Meta-programming helper classes

#ifndef L3STER_UTIL_META_HPP
#define L3STER_UTIL_META_HPP

#include "l3ster/util/Common.hpp"
#include "l3ster/util/Concepts.hpp"
#include "l3ster/util/StaticVector.hpp"

#include <functional>
#include <numeric>
#include <utility>
#include <variant>

namespace lstr::util
{
namespace detail
{
template < std::integral T, T first, T last >
constexpr auto makeIntervalArray()
    requires(first <= last)
{
    std::array< T, last - first + 1 > interval;
    std::ranges::iota(interval, first);
    return interval;
}

template < std::array A >
constexpr auto intSeqFromArray()
    requires std::integral< typename decltype(A)::value_type >
{
    return []< size_t... I >(std::index_sequence< I... >) {
        return std::integer_sequence< typename decltype(A)::value_type, A[I]... >{};
    }(std::make_index_sequence< A.size() >{});
}
} // namespace detail

template < std::integral T, T first, T last >
    requires(first <= last)
using int_seq_interval = decltype(detail::intSeqFromArray< detail::makeIntervalArray< T, first, last >() >());

namespace detail
{
// Avoid re-calculating the cartesian product for different Inner/Outer templates
template < std::ranges::sized_range auto... Params >
struct CartesianProductHolder
{
    static constexpr size_t size = (std::ranges::size(Params) * ...);
    static consteval auto   makeCartSoA()
    {
        auto cart_prod_view = std::views::cartesian_product(Params...);
        auto retval         = std::tuple< std::array< std::ranges::range_value_t< decltype(Params) >, size >... >{};
        [&]< size_t... I >(std::index_sequence< I... >) {
            (std::ranges::copy(cart_prod_view | std::views::elements< I >, std::get< I >(retval).begin()), ...);
        }(std::make_index_sequence< sizeof...(Params) >{});
        return retval;
    }
    static constexpr auto cart_prod_soa = makeCartSoA();
};

template < template < auto... > typename Inner,
           template < typename... > typename Outer,
           std::ranges::sized_range auto... Params >
class CartesianProductApplyImpl
{
    template < size_t I >
    using getInner = decltype([]< size_t... Pi >(std::index_sequence< Pi... >) {
        return Inner< (std::get< Pi >(CartesianProductHolder< Params... >::cart_prod_soa).at(I))... >{};
    }(std::make_index_sequence< sizeof...(Params) >{}));

public:
    using type = decltype([]< size_t... I >(std::index_sequence< I... >) {
        return Outer< getInner< I >... >{};
    }(std::make_index_sequence< CartesianProductHolder< Params... >::size >{}));
};
} // namespace detail

template < template < auto... > typename Inner,
           template < typename... > typename Outer,
           std::ranges::range auto... Params >
    requires((not std::ranges::empty(Params) and ...))
using CartesianProductApply = detail::CartesianProductApplyImpl< Inner, Outer, Params... >::type;

template < size_t N >
constexpr auto getTrueInds(const std::array< bool, N >& a) -> util::StaticVector< size_t, N >
{
    auto retval = util::StaticVector< size_t, N >{};
    std::ranges::copy_if(std::views::iota(0uz, a.size()), std::back_inserter(retval), [&a](size_t i) { return a[i]; });
    return retval;
}

template < size_t N >
constexpr auto getTrueInds(const std::bitset< N >& a) -> util::StaticVector< size_t, N >
{
    auto retval = util::StaticVector< size_t, N >{};
    std::ranges::copy_if(std::views::iota(0uz, a.size()), std::back_inserter(retval), [&a](size_t i) { return a[i]; });
    return retval;
}

template < ArrayOf_c< bool > auto A >
constexpr auto getTrueInds(ConstexprValue< A > = {})
{
    constexpr auto true_inds_sv = getTrueInds(A);
    auto           retval       = std::array< size_t, true_inds_sv.size() >{};
    std::ranges::copy(true_inds_sv, retval.begin());
    return retval;
}

template < auto... >
inline constexpr bool always_false = false;
} // namespace lstr::util
#endif // L3STER_UTIL_META_HPP
