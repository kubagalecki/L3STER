#ifndef L3STER_UTIL_RANGES_HPP
#define L3STER_UTIL_RANGES_HPP

#include "l3ster/util/ArrayOwner.hpp"
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

template < std::ranges::range Range >
using range_const_reference_t = std::add_lvalue_reference_t< std::add_const_t< std::ranges::range_value_t< Range > > >;

template < std::ranges::range... Ranges >
auto gatherAsCommon(Ranges&&... ranges)
    requires requires { typename std::common_type< std::ranges::range_value_t< Ranges >... >::type; }
{
    using common_t = std::common_type_t< std::ranges::range_value_t< Ranges >... >;
    auto retval    = util::ArrayOwner< util::ArrayOwner< common_t > >(sizeof...(Ranges));
    auto push_copy = [&retval, i = 0]< typename R >(R&& r) mutable {
        retval[i++] = std::forward< R >(r) | std::views::all;
    };
    (push_copy(std::forward< Ranges >(ranges)), ...);
    return retval;
}

template < std::ranges::range... Ranges >
auto concatRanges(Ranges&&... ranges)
    requires requires { typename std::common_type< std::ranges::range_value_t< Ranges >... >::type; }
{
    using common_t        = std::common_type_t< std::ranges::range_value_t< Ranges >... >;
    const auto size       = static_cast< size_t >((std::ranges::distance(ranges) + ...));
    auto       retval     = ArrayOwner< common_t >(size);
    auto       write_copy = [iter = retval.begin()]< typename R >(R&& r) mutable {
        iter = std::ranges::copy(std::forward< R >(r), iter).out;
    };
    (write_copy(std::forward< Ranges >(ranges)), ...);
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_RANGES_HPP
