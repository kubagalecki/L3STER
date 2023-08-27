#ifndef L3STER_UTIL_RANGES_HPP
#define L3STER_UTIL_RANGES_HPP

#include "l3ster/util/Common.hpp"

#include <algorithm>
#include <ranges>

namespace lstr::util
{
template < IndexRange_c IndexRange >
bool isValidIndexRange(IndexRange&& inds, size_t size, std::source_location sl = std::source_location::current())
{
    return std::ranges::all_of(std::forward< IndexRange >(inds),
                               [=](auto i) { return exactIntegerCast< size_t >(i, sl) < size; });
}

template < typename Range, typename Inds >
auto makeIndexedView(Range&& range, Inds&& inds)
    requires std::ranges::viewable_range< Range > and std::ranges::random_access_range< Range > and
             std::ranges::viewable_range< Inds > and std::integral< std::ranges::range_value_t< Inds > >
{
    return std::forward< Inds >(inds) |
           std::views::transform([r = std::forward< Range >(range) | std::views::all](auto i) { return r.begin()[i]; });
}

template < typename T, std::ranges::range R >
decltype(auto) castView(R&& range)
    requires std::convertible_to< std::ranges::range_value_t< R >, T >
{
    if constexpr (std::same_as< std::ranges::range_value_t< R >, T >)
        return std::forward< R >(range);
    else
        return std::forward< R >(range) | std::views::transform([](const auto& in) { return static_cast< T >(in); });
}

template < std::ranges::range R >
auto toVector(R&& range) -> std::vector< std::ranges::range_value_t< R > >
{
    auto common = std::forward< R >(range) | std::views::common;
    return {common.begin(), common.end()};
}

template < std::ranges::range V >
decltype(auto) toVector(V&& vector)
    requires Vector_c< V >
{
    return std::forward< V >(vector);
}
} // namespace lstr::util
#endif // L3STER_UTIL_RANGES_HPP
