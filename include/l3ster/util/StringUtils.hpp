#ifndef L3STER_UTIL_STRINGUTILS_HPP
#define L3STER_UTIL_STRINGUTILS_HPP

#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/Concepts.hpp"

#include <cctype>
#include <charconv>
#include <string_view>

namespace lstr::util
{
template < Arithmetic_c T >
auto fromString(std::string_view str) -> T
{
    const auto begin  = str.data();
    const auto end    = std::next(begin, std::ranges::ssize(str));
    auto       retval = T{};
    util::throwingAssert(std::from_chars(begin, end, retval).ec == std::errc{}, "Failed string conversion");
    return retval;
}

inline bool equalCaseInsensitive(std::string_view s1, std::string_view s2)
{
    constexpr auto to_lower = [](char c) {
        return static_cast< char >(std::tolower(static_cast< unsigned char >(c)));
    };
    return std::ranges::equal(s1, s2, {}, to_lower, to_lower);
}
} // namespace lstr::util
#endif // L3STER_UTIL_STRINGUTILS_HPP
