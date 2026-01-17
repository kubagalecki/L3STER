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

template < typename T, size_t... Ns >
auto concatArrays(const std::array< T, Ns >&... arrays) -> std::array< T, (Ns + ...) >
{
    auto retval      = std::array< T, (Ns + ...) >{};
    auto write_array = [out = retval.begin()](const auto& a) mutable {
        out = std::ranges::copy(a, out).out;
    };
    (write_array(arrays), ...);
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_RANGES_HPP
