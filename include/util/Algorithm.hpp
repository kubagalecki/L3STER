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

template < typename T, size_t SIZE, size_t TRIMMED_SIZE >
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
template < template < typename > typename Predicate, tuple Aggregate, tuple_like Appended >
constexpr auto appendTupleIf(Aggregate&& aggregate, Appended&& appended)
{
    if constexpr (Predicate< Appended >::value)
        return std::tuple_cat(std::forward< Aggregate >(aggregate),
                              std::make_tuple(std::forward< Appended >(appended)));
    else
        return std::forward< Aggregate >(aggregate);
}

template < template < typename > typename, tuple T >
struct ConditionAndTuple
{
    T tuple;
};

template < template < typename > typename Predicate, tuple T, tuple_like A >
constexpr auto operator<<(const ConditionAndTuple< Predicate, T >& aggregate, A&& appended)
{
    return ConditionAndTuple< Predicate, decltype(appendTupleIf< Predicate >(aggregate.tuple, appended)) >{
        appendTupleIf< Predicate >(aggregate.tuple, appended)};
}
} // namespace detail

template < template < typename > typename Predicate, predicate_trait_specialized< Predicate >... T >
constexpr auto tuplifyIf(T&&... tup)
{
    detail::ConditionAndTuple< Predicate, std::tuple<> > empty;
    return (empty << ... << std::forward< T >(tup)).tuple;
}
} // namespace lstr
#endif // L3STER_UTIL_ALGORITHM_HPP
