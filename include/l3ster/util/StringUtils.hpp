#ifndef L3STER_UTILS_STRINGUTILS_HPP
#define L3STER_UTILS_STRINGUTILS_HPP

#include <algorithm>
#include <ranges>
#include <string>

namespace lstr
{
std::string prependSpaces(const std::string& str, std::size_t desired_len)
{
    std::string retval(desired_len, ' ');
    std::ranges::copy(str | std::views::reverse | std::views::take(desired_len), retval.rbegin());
    return retval;
}
} // namespace lstr
#endif // L3STER_UTILS_STRINGUTILS_HPP
