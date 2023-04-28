#ifndef L3STER_UTIL_TBBUTILS_HPP
#define L3STER_UTIL_TBBUTILS_HPP

#include "l3ster/util/Concepts.hpp"

#include "oneapi/tbb.h"

#include <iterator>
#include <ranges>

namespace lstr::util::tbb
{
namespace detail
{
template < std::ranges::sized_range Range >
using blocked_iter_space_t = oneapi::tbb::blocked_range< std::ranges::range_difference_t< Range > >;

auto makeBlockedIterSpace(std::ranges::sized_range auto&& range) -> blocked_iter_space_t< decltype(range) >
{
    return {0, std::ranges::ssize(range)};
}
} // namespace detail

void parallelFor(SizedRandomAccessRange_c auto&&                                            range,
                 std::invocable< std::ranges::range_reference_t< decltype(range) > > auto&& kernel)
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    using iter_space_t    = decltype(iter_space);
    oneapi::tbb::parallel_for(iter_space,
                              [&kernel, range_begin = std::ranges::begin(range)](const iter_space_t& subrange) {
                                  for (auto i : std::views::iota(subrange.begin(), subrange.end()))
                                      std::invoke(kernel, range_begin[i]);
                              });
}

template < typename Reduction = std::plus<>, typename Transform = std::identity >
auto parallelTransformReduce(SizedRandomAccessRange_c auto&& range,
                             const auto&                     identity,
                             Reduction&&                     reduction = {},
                             Transform&&                     transform = {}) -> std::decay_t< decltype(identity) >
    requires requires(std::add_lvalue_reference_t< std::add_const_t< std::ranges::range_value_t< decltype(range) > > >
                          range_element) {
        {
            std::invoke(transform, range_element)
        } -> std::convertible_to< decltype(identity) >;
        {
            std::invoke(reduction, identity, identity)
        } -> std::convertible_to< decltype(identity) >;
        {
            std::transform_reduce(std::ranges::cbegin(range), std::ranges::cend(range), identity, reduction, transform)
        } -> std::convertible_to< decltype(identity) >;
    }
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    return oneapi::tbb::parallel_reduce(
        iter_space,
        identity,
        [&](const auto& iter_range, const auto& value) {
            return std::transform_reduce(std::execution::unseq,
                                         std::next(range.begin(), iter_range.begin()),
                                         std::next(range.begin(), iter_range.end()),
                                         value,
                                         reduction,
                                         transform);
        },
        reduction);
}
} // namespace lstr::util::tbb
#endif // L3STER_UTIL_TBBUTILS_HPP
