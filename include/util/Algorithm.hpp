#ifndef L3STER_UTIL_ALGORITHM_HPP
#define L3STER_UTIL_ALGORITHM_HPP
#include "util/Concepts.hpp"

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
        } -> std::convertible_to< bool >;
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
        } -> std::convertible_to< std::ptrdiff_t >;
}

void copyPermuted(It_in first_in, It_in last_in, It_perm first_perm, It_out first_out)
{
    for (auto i = std::distance(first_in, last_in); i > 0; --i)
        *first_out++ = first_in[*first_perm++];
}

template < typename T, size_t N >
constexpr std::array< T, N > getSortedArray(const std::array< T, N >& array)
{
    auto sorted = array;
    std::ranges::sort(sorted);
    return sorted;
}

template < size_t TRIMMED_SIZE, typename T, size_t SIZE >
requires(SIZE >= TRIMMED_SIZE) constexpr auto trimArray(const std::array< T, SIZE >& a)
{
    if constexpr (SIZE == TRIMMED_SIZE)
        return a;
    else
    {
        std::array< T, TRIMMED_SIZE > trimmed;
        std::copy(a.cbegin(), a.cbegin() + TRIMMED_SIZE, trimmed.begin());
        return trimmed;
    }
}

namespace detail
{
template < template < typename > typename TraitsPredicate, typename T >
constexpr auto tuplifyIf(T&& arg)
{
    if constexpr (TraitsPredicate< std::decay_t< T > >::value)
        return std::make_tuple(std::forward< T >(arg));
    else
        return std::tuple<>{};
}
} // namespace detail

template < template < typename > typename TraitsPredicate, predicate_trait_specialized< TraitsPredicate >... T >
constexpr auto makeTupleIf(T&&... arg)
{
    return std::tuple_cat(detail::tuplifyIf< TraitsPredicate >(std::forward< T >(arg))...);
}
} // namespace lstr
#endif // L3STER_UTIL_ALGORITHM_HPP
