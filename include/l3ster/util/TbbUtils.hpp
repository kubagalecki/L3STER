#ifndef L3STER_UTIL_TBBUTILS_HPP
#define L3STER_UTIL_TBBUTILS_HPP

#include "l3ster/util/Concepts.hpp"
#include "l3ster/util/Ranges.hpp"

#include "oneapi/tbb.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <iterator>
#include <map>
#include <ranges>
#include <shared_mutex>

namespace lstr::util::tbb
{
namespace detail
{
template < std::ranges::sized_range Range >
using blocked_iter_space_t = oneapi::tbb::blocked_range< std::ranges::range_difference_t< Range > >;

template < std::ranges::sized_range Range >
auto makeBlockedIterSpace(Range&& range, size_t grain = 1) -> blocked_iter_space_t< Range >
{
    return {0, std::ranges::ssize(range), grain};
}
} // namespace detail

template < SizedRandomAccessRange_c Range, std::invocable< std::ranges::range_reference_t< Range > > Kernel >
void parallelFor(Range&& range, Kernel&& kernel)
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    using iter_space_t    = decltype(iter_space);
    oneapi::tbb::parallel_for(iter_space,
                              [&kernel, range_begin = std::ranges::begin(range)](const iter_space_t& subrange) {
                                  for (auto i : std::views::iota(subrange.begin(), subrange.end()))
                                      std::invoke(kernel, range_begin[i]);
                              });
}

template < std::ranges::sized_range Range, std::invocable< std::ranges::range_reference_t< Range > > Kernel >
void parallelFor(Range&& range, Kernel&& kernel)
{
    auto&& range_common = std::forward< Range >(range) | std::views::common;
    auto   begin = std::ranges::begin(range_common), end = std::ranges::end(range_common);
    oneapi::tbb::parallel_for_each(begin, end, std::forward< Kernel >(kernel));
}

template < SizedRandomAccessRange_c Range, std::random_access_iterator Iter, typename Kernel >
void parallelTransform(Range&& input_range, Iter output_iter, Kernel&& kernel)
    requires std::invocable< Kernel, range_const_reference_t< Range > > and
             std::output_iterator< Iter, std::invoke_result_t< Kernel, range_const_reference_t< Range > > >
{
    const auto iter_space  = detail::makeBlockedIterSpace(input_range);
    using iter_space_t     = decltype(iter_space);
    const auto range_begin = std::ranges::cbegin(input_range);
    oneapi::tbb::parallel_for(iter_space, [&](const iter_space_t& subrange) {
        for (auto i : std::views::iota(subrange.begin(), subrange.end()))
            output_iter[i] = std::invoke(kernel, range_begin[i]);
    });
}

template < typename Value, typename Zero, typename Reduction, typename Transform >
concept TransformReducible_c =
    std::is_invocable_r_v< Zero, Transform, Value > and std::is_invocable_r_v< Zero, Reduction, Zero, Zero >;

template < SizedRandomAccessRange_c Range,
           typename Zero,
           typename Reduction = std::plus<>,
           typename Transform = std::identity >
auto parallelTransformReduce(Range&& range, Zero identity, Reduction reduction = {}, Transform transform = {})
    -> std::decay_t< Zero >
    requires TransformReducible_c< range_const_reference_t< Range >, Zero, Reduction, Transform >
{
    const auto iter_space = detail::makeBlockedIterSpace(range);
    return oneapi::tbb::parallel_reduce(
        iter_space,
        identity,
        [&](const auto& iter_range, const auto& value) {
            return std::transform_reduce(std::next(range.begin(), iter_range.begin()),
                                         std::next(range.begin(), iter_range.end()),
                                         value,
                                         reduction,
                                         transform);
        },
        reduction);
}
} // namespace lstr::util::tbb
#endif // L3STER_UTIL_TBBUTILS_HPP
