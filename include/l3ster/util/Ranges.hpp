#ifndef L3STER_UTIL_RANGES_HPP
#define L3STER_UTIL_RANGES_HPP

#include "l3ster/util/Common.hpp"

#include <algorithm>
#include <ranges>

namespace lstr::util
{
bool isValidIndexRange(IndexRange_c auto&& inds, size_t size, std::source_location sl = std::source_location::current())
{
    return std::ranges::all_of(std::forward< decltype(inds) >(inds),
                               [=](auto i) { return exactIntegerCast< size_t >(i, sl) < size; });
}

auto makeIndexedView(auto&& range, auto&& inds)
    requires std::ranges::viewable_range< decltype(range) > and std::ranges::random_access_range< decltype(range) > and
             std::ranges::viewable_range< decltype(inds) > and
             std::integral< std::ranges::range_value_t< decltype(inds) > >
{
    return std::forward< decltype(inds) >(inds) |
           std::views::transform(
               [r = std::forward< decltype(range) >(range) | std::views::all](auto i) { return r.begin()[i]; });
}
} // namespace lstr::util
#endif // L3STER_UTIL_RANGES_HPP
