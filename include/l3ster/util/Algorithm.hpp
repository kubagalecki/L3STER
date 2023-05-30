#ifndef L3STER_UTIL_ALGORITHM_HPP
#define L3STER_UTIL_ALGORITHM_HPP

#include "l3ster/util/Meta.hpp"

#include "oneapi/tbb/parallel_invoke.h"

#include <iterator>
#include <numeric>
#include <vector>

namespace lstr::util
{
template < std::random_access_iterator It, typename F >
std::vector< size_t > sortingPermutation(It first, It last, F&& compare)
    requires requires(It i, F f) {
        {
            f(*i, *i)
        } -> std::convertible_to< bool >;
    }
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
void copyPermuted(It_in first_in, It_in last_in, It_perm first_perm, It_out first_out)
    requires requires(It_perm it) {
        {
            *it
        } -> std::convertible_to< std::ptrdiff_t >;
    }
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

template < size_t trimmed_size, typename T, size_t size >
constexpr auto trimArray(const std::array< T, size >& a)
    requires(size >= trimmed_size)
{
    if constexpr (size == trimmed_size)
        return a;
    else
    {
        std::array< T, trimmed_size > trimmed;
        std::copy(a.cbegin(), a.cbegin() + trimmed_size, trimmed.begin());
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

constexpr void forEachTuple(auto&& t, auto&& f)
    requires tuple_like< std::decay_t< decltype(t) > > and tuple_invocable< decltype(f), std::decay_t< decltype(t) > >
{
    std::invoke([&]< size_t... I >(std::index_sequence< I... >) { (std::invoke(f, std::get< I >(t)), ...); },
                std::make_index_sequence< std::tuple_size_v< std::decay_t< decltype(t) > > >{});
}

template < std::unsigned_integral T >
std::vector< T > consecutiveIndices(T n)
{
    std::vector< T > retval(n);
    std::iota(begin(retval), end(retval), T{0});
    return retval;
}

template < std::unsigned_integral T, T N >
constexpr std::array< T, N > consecutiveIndices(std::integral_constant< T, N >)
{
    std::array< T, N > retval;
    std::iota(begin(retval), end(retval), T{0});
    return retval;
}

template < std::copy_constructible T_a, std::integral T_filter, size_t N_a, size_t N_filter >
constexpr auto arrayAtInds(const std::array< T_a, N_a >& array, const std::array< T_filter, N_filter >& filter)
{
    std::array< T_a, N_filter > retval;
    std::ranges::transform(filter, begin(retval), [&](T_filter i) { return array[i]; });
    return retval;
}

template < std::ranges::range        R1,
           std::ranges::range        R2,
           std::weakly_incrementable O,
           typename Pred  = std::ranges::equal_to,
           typename Proj1 = std::identity,
           typename Proj2 = std::identity >
constexpr auto
matchingPermutation(R1&& r_pattern, R2&& r_match, O out, Pred pred = {}, Proj1 proj1 = {}, Proj2 proj2 = {})
    requires std::indirectly_writable< O, size_t > and std::indirectly_comparable< std::ranges::iterator_t< R1 >,
                                                                                   std::ranges::iterator_t< R2 >,
                                                                                   Pred,
                                                                                   Proj1,
                                                                                   Proj2 >
{
    auto out_initial = out;
    std::ranges::for_each(r_match, [&](auto&& match_el) {
        *(out++) = std::distance(std::ranges::begin(r_pattern),
                                 std::ranges::find_if(
                                     r_pattern, [&](auto&& el) { return pred(el, proj2(match_el)); }, proj1));
    });
    return std::ranges::in_out_result{out_initial, out};
}

template < typename F, std::integral T, T... I >
constexpr void forConstexpr(F&& f, std::integer_sequence< T, I... >)
    requires(std::invocable< F, std::integral_constant< T, I > > and ...)
{
    (f(std::integral_constant< T, I >{}), ...);
}

template < typename F, std::ranges::sized_range auto R >
constexpr void forConstexpr(F&& f, ConstexprValue< R >)
{
    using diff_t                = std::ranges::range_difference_t< decltype(R) >;
    constexpr auto access_range = [](diff_t index) {
        return *std::next(std::ranges::cbegin(R), index);
    };
    forConstexpr([&]< diff_t I >(std::integral_constant< diff_t, I >) { f(ConstexprValue< access_range(I) >{}); },
                 std::make_integer_sequence< diff_t, std::ranges::size(R) >{});
}

template < typename F, std::ranges::range auto R >
void forEachConstexprParallel(F&& f, ConstexprValue< R >)
{
    if constexpr (std::ranges::distance(R) > 1) // tbb::parallel_invoke requires at least 2 function objects
    {
        using diff_t                 = std::ranges::range_difference_t< decltype(R) >;
        const auto invoke_on_indices = [&f]< diff_t... I >(std::integer_sequence< diff_t, I... >) {
            constexpr auto access_range = [](diff_t index) {
                return *std::next(std::ranges::cbegin(R), index);
            };
            oneapi::tbb::parallel_invoke([&] {
                f(ConstexprValue< access_range(I) >{});
            }...);
        };
        invoke_on_indices(std::make_integer_sequence< diff_t, std::ranges::distance(R) >{});
    }
    else if constexpr (std::ranges::distance(R) == 1)
        std::invoke(std::forward< F >(f), ConstexprValue< *std::ranges::cbegin(R) >{});
    else
        return;
}

template < typename T >
constexpr bool contains(std::initializer_list< T > list, T value)
{
    return std::ranges::any_of(list, [value = value](T t) { return t == value; });
}

template <
    std::ranges::forward_range                                                                   R,
    std::indirect_binary_predicate< std::ranges::iterator_t< R >, std::ranges::iterator_t< R > > Cmp =
        std::ranges::equal_to,
    std::regular_invocable< std::ranges::range_value_t< R >, std::ranges::range_value_t< R > > Red = std::plus<> >
constexpr std::ranges::borrowed_subrange_t< R >
reduceConsecutive(R&& range, Cmp&& comparator = {}, Red&& reduction = {})
    requires std::permutable< std::ranges::iterator_t< R > > and
             std::assignable_from<
                 std::ranges::range_reference_t< R >,
                 std::invoke_result_t< Red, std::ranges::range_value_t< R >, std::ranges::range_value_t< R > > >
{
    auto it        = std::ranges::begin(range);
    auto write_pos = it;
    while (it != std::ranges::end(range))
    {
        const auto adj_range_begin = std::adjacent_find(it, std::ranges::end(range), comparator);
        if (it != write_pos)
            std::copy(it, adj_range_begin, write_pos);
        std::advance(write_pos, std::distance(it, adj_range_begin));
        if (adj_range_begin == std::ranges::end(range))
            break;
        auto next_adjacent = std::next(adj_range_begin);
        *write_pos         = std::invoke(reduction, std::as_const(*adj_range_begin), std::as_const(*next_adjacent));
        while (std::next(next_adjacent) != std::ranges::end(range) and
               std::invoke(comparator, std::as_const(*next_adjacent), std::as_const(*std::next(next_adjacent))))
            *write_pos = reduction(*write_pos, *++next_adjacent);
        ++write_pos;
        it = ++next_adjacent;
    }
    return std::ranges::borrowed_subrange_t< R >(write_pos, std::ranges::end(range));
}

template < typename T, size_t size >
constexpr auto makeIotaArray(const T& first = T{})
{
    std::array< T, size > retval;
    std::iota(begin(retval), end(retval), first);
    return retval;
}

template < typename T >
void sortRemoveDup(std::vector< T >& vec)
{
    std::ranges::sort(vec);
    const auto erase_range = std::ranges::unique(vec);
    vec.erase(erase_range.begin(), erase_range.end());
    vec.shrink_to_fit();
}

template < std::array array >
constexpr auto sortedUniqueElements()
{
    const auto n_unique = std::invoke([] {
        auto a = array;
        std::ranges::sort(a);
        return static_cast< std::size_t >(std::distance(a.begin(), std::ranges::unique(a).begin()));
    });
    auto       retval   = std::array< typename decltype(array)::value_type, n_unique >{};
    std::ranges::unique_copy(array, retval.begin());
    return retval;
}

template < IndexRange_c auto inds, typename T, size_t N, std::indirectly_writable< T > Iter >
Iter copyValuesAtInds(const std::array< T, N >& array, Iter out_iter, ConstexprValue< inds > = {})
    requires(std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    for (auto i : inds)
        *out_iter++ = array[i];
    return out_iter;
}

template < IndexRange_c auto inds, typename T, size_t N >
std::array< T, std::ranges::size(inds) > getValuesAtInds(const std::array< T, N >& array,
                                                         ConstexprValue< inds >    inds_ctwrpr = {})
    requires(std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    std::array< T, std::ranges::size(inds) > retval;
    copyValuesAtInds(array, begin(retval), inds_ctwrpr);
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_ALGORITHM_HPP
