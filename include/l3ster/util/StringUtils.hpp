#ifndef L3STER_UTIL_STRINGUTILS_HPP
#define L3STER_UTIL_STRINGUTILS_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <charconv>
#include <string_view>

namespace lstr::util
{
template < Arithmetic_c T >
auto fromString(std::string_view str)
{
    const auto begin  = str.data();
    const auto end    = std::next(begin, std::ranges::ssize(str));
    auto       retval = T{};
    util::throwingAssert(std::from_chars(begin, end, retval).ec == std::errc{}, "Failed string conversion");
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_STRINGUTILS_HPP
