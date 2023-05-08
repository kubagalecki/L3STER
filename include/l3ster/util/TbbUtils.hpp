#ifndef L3STER_UTIL_TBBUTILS_HPP
#define L3STER_UTIL_TBBUTILS_HPP

#include "l3ster/util/Concepts.hpp"

#include "oneapi/tbb.h"

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

void parallelFor(std::ranges::sized_range auto&&                                            range,
                 std::invocable< std::ranges::range_reference_t< decltype(range) > > auto&& kernel)
{
    using diff_t     = std::ranges::range_difference_t< decltype(range) >;
    auto iter_cache  = std::map< diff_t, std::ranges::iterator_t< decltype(range) > >{};
    auto cache_mutex = std::shared_mutex{};
    iter_cache.emplace(0, std::ranges::begin(range));
    iter_cache.emplace(std::ranges::ssize(range), std::ranges::end(range));

    const auto get_iter = [&](std::ranges::range_difference_t< decltype(range) > index) {
        const auto [closest_ind, closest_iter] = std::invoke([&] {
            const auto lock = std::shared_lock{cache_mutex};
            return *std::prev(iter_cache.upper_bound(index));
        });
        return std::next(closest_iter, index - closest_ind);
    };
    const auto cache_iter = [&](std::ranges::range_difference_t< decltype(range) > index,
                                std::ranges::iterator_t< decltype(range) >         iter) {
        {
            const auto lock = std::shared_lock{cache_mutex};
            if (iter_cache.contains(index))
                return;
        }
        const auto lock = std::lock_guard{cache_mutex};
        iter_cache.emplace(index, iter);
    };

    const auto iter_space = detail::makeBlockedIterSpace(range);
    using iter_space_t    = std::remove_const_t< decltype(iter_space) >;
    oneapi::tbb::parallel_for(iter_space, [&](const iter_space_t& subrange) {
        auto       ind     = subrange.begin();
        auto       iter    = get_iter(ind);
        const auto end_ind = subrange.end();
        for (; ind != end_ind; ++ind)
            std::invoke(kernel, *iter++);
        cache_iter(ind, iter);
    });
}

void parallelTransform(SizedRandomAccessRange_c auto&& input_range, auto output_iter, auto&& kernel)
    requires std::random_access_iterator< decltype(output_iter) > and
             requires(decltype(*std::ranges::cbegin(input_range)) input_element) {
                 *output_iter = std::invoke(kernel, input_element);
             }
{
    const auto iter_space = detail::makeBlockedIterSpace(input_range);
    using iter_space_t    = decltype(iter_space);
    oneapi::tbb::parallel_for(iter_space,
                              [&, range_begin = std::ranges::cbegin(input_range)](const iter_space_t& subrange) {
                                  for (auto i : std::views::iota(subrange.begin(), subrange.end()))
                                      output_iter[i] = std::invoke(kernel, range_begin[i]);
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
