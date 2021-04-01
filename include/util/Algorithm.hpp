#ifndef L3STER_UTIL_ALGORITHM_HPP
#define L3STER_UTIL_ALGORITHM_HPP

#include <iterator>
#include <numeric>
#include <vector>

namespace lstr
{
template < std::random_access_iterator It, typename F >
requires requires(It i, F f)
{
    {
        f(*i, *i)
    }
    ->std::convertible_to< bool >;
}
std::vector< size_t > sortingPermutation(It first, It last, F&& compare)
{
    std::vector< size_t > indices(std::distance(first, last));
    std::iota(begin(indices), end(indices), 0u);
    std::sort(begin(indices), end(indices), [&](size_t a, size_t b) { return compare(first[a], first[b]); });
    return indices;
}
template < std::random_access_iterator It >
std::vector< size_t > sortingPermutation(It first, It last)
{
    return sortingPermutation(first, last, std::less< decltype(*first) >{});
}

template < std::random_access_iterator                                It_in,
           std::input_iterator                                        It_perm,
           std::output_iterator< decltype(*std::declval< It_in >()) > It_out >
requires requires(It_perm it)
{
    {
        *it
    }
    ->std::convertible_to< std::ptrdiff_t >;
}
void copyPermuted(It_in first_in, It_in last_in, It_perm first_perm, It_out first_out)
{
    for (auto i = std::distance(first_in, last_in); i > 0; --i)
        *first_out++ = first_in[*first_perm++];
}
} // namespace lstr
#endif // L3STER_UTIL_ALGORITHM_HPP
