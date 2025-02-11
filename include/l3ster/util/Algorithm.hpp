#ifndef L3STER_UTIL_ALGORITHM_HPP
#define L3STER_UTIL_ALGORITHM_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/CrsGraph.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/Meta.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include "oneapi/tbb/parallel_invoke.h"

#include <iterator>
#include <numeric>
#include <vector>

namespace lstr::util
{
template < std::random_access_iterator It, typename F >
std::vector< size_t > sortingPermutation(It first, It last, F&& compare)
    requires requires(It i, F f) {
        { f(*i, *i) } -> std::convertible_to< bool >;
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
        { *it } -> std::convertible_to< std::ptrdiff_t >;
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

template < typename Tup, typename F >
constexpr void forEachTuple(Tup&& t, F&& f)
    requires tuple_like< std::remove_cvref_t< Tup > > and tuple_invocable< F, std::remove_cvref_t< Tup > >
{
    const auto visit_inds = [&]< size_t... I >(std::index_sequence< I... >) {
        (std::invoke(f, std::get< I >(t)), ...);
    };
    std::invoke(visit_inds, std::make_index_sequence< std::tuple_size_v< std::remove_cvref_t< Tup > > >{});
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
        *(out++) =
            std::distance(std::ranges::begin(r_pattern),
                          std::ranges::find_if(r_pattern, [&](auto&& el) { return pred(el, proj2(match_el)); }, proj1));
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

template < typename T >
auto sortRemoveDup(util::ArrayOwner< T >& vec) -> size_t
{
    std::ranges::sort(vec);
    const auto erase_range = std::ranges::unique(vec);
    return static_cast< size_t >(std::distance(vec.begin(), erase_range.begin()));
}

template < typename T >
auto getUniqueCopy(util::ArrayOwner< T > vec) -> util::ArrayOwner< T >
{
    const auto unique_sz = sortRemoveDup(vec);
    return unique_sz == vec.size() ? vec : util::ArrayOwner< T >{vec | std::views::take(unique_sz)};
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
auto getValuesAtInds(const std::array< T, N >& array, ConstexprValue< inds > inds_ctwrpr = {})
    -> std::array< T, std::ranges::size(inds) >
    requires(std::ranges::all_of(inds, [](size_t i) { return i < N; }))
{
    std::array< T, std::ranges::size(inds) > retval;
    copyValuesAtInds(array, begin(retval), inds_ctwrpr);
    return retval;
}

template < typename T, typename Process >
void staggeredAllGather(const MpiComm& comm, std::span< const T > my_data, Process&& process_received)
    requires std::invocable< Process, std::span< const T >, int >
{
    const int my_rank   = comm.getRank();
    const int comm_size = comm.getSize();

    // Data is not actually modified, this is just to enable passing to the MPI wrapper
    const auto my_data_mut = std::span{const_cast< T* >(my_data.data()), my_data.size()};

    auto msg_sizes = util::ArrayOwner< size_t >(static_cast< size_t >(comm_size));
    comm.allGather(std::views::single(my_data.size()), msg_sizes.begin());
    msg_sizes[static_cast< size_t >(my_rank)] = 0;
    const auto            max_recv_sz         = std::ranges::max(msg_sizes);
    util::ArrayOwner< T > proc_buf(max_recv_sz), recv_buf(max_recv_sz);

    auto       request            = MpiComm::Request{};
    const auto begin_broadcasting = [&](int send_rank) {
        const auto comm_buf =
            my_rank == send_rank ? my_data_mut : std::span{recv_buf}.first(msg_sizes[static_cast< size_t >(send_rank)]);
        request = comm.broadcastAsync(comm_buf, send_rank);
    };
    const auto finish_broadcasting = [&]() {
        request.wait();
        std::swap(recv_buf, proc_buf);
    };
    const auto do_processing = [&](int send_rank) {
        const auto proc_data = my_rank == send_rank ? my_data : std::span{proc_buf.cbegin(), msg_sizes[send_rank]};
        std::invoke(process_received, proc_data, send_rank);
    };

    begin_broadcasting(0);
    for (int send_rank = 1; send_rank != comm_size; ++send_rank)
    {
        finish_broadcasting();
        begin_broadcasting(send_rank);
        do_processing(send_rank - 1);
    }
    finish_broadcasting();
    do_processing(comm_size - 1);
}

// CRS graph algorithms
template < typename VertexType, typename Fun >
auto depthFirstSearch(const CrsGraph< VertexType >& graph, Fun&& op, VertexType start = 0)
    -> robin_hood::unordered_flat_set< VertexType >
    requires std::invocable< Fun, VertexType >
{
    auto visited  = robin_hood::unordered_flat_set< VertexType >{};
    auto to_visit = std::vector< VertexType >{start};
    while (not to_visit.empty())
    {
        const auto current = to_visit.back();
        to_visit.pop_back();
        if (not visited.contains(current))
        {
            std::invoke(op, current);
            visited.insert(current);
            std::ranges::remove_copy_if(graph(current) | std::views::reverse,
                                        std::back_inserter(to_visit),
                                        [&](VertexType vertex) { return visited.contains(vertex); });
        }
    }
    return visited;
}

/// Get components of an undirected graph. Assumes $v_1 \in graph(v_2) <=> v_2 \in graph(v_1)$.
/// Each element of the returned vector is the sorted set of vertices constituting a component of the input graph
template < typename VertexType >
auto getComponentsUndirected(const CrsGraph< VertexType >& graph) -> std::vector< ArrayOwner< VertexType > >
{
    constexpr auto ignore = [](VertexType) {
    };
    auto visited_lu = DynamicBitset{graph.getNRows()};
    auto retval     = std::vector< ArrayOwner< VertexType > >{};
    for (VertexType v = 0; v != graph.getNRows(); ++v)
    {
        if (visited_lu.test(v))
            continue;
        retval.emplace_back(depthFirstSearch(graph, ignore, v));
        std::ranges::sort(retval.back());
        for (auto i : retval.back())
            visited_lu.set(i);
    }
    return retval;
}

enum struct GraphType
{
    Undirected,
    Directed
};

/// Creates CRS graph from edges, defined by pairs of corresponding elements of the 2 input ranges
/// If graph_type == GraphType::Undirected, both (from[i], to[i]) and (to[i], from[i]) will be included in the result
template < std::ranges::range From, std::ranges::range To >
auto makeCrsGraph(From&& from, To&& to, GraphType graph_type = GraphType::Directed)
    requires std::integral< std::remove_cvref_t< std::ranges::range_value_t< From > > > and
             std::integral< std::remove_cvref_t< std::ranges::range_value_t< To > > > and
             std::common_with< std::remove_cvref_t< std::ranges::range_value_t< From > >,
                               std::remove_cvref_t< std::ranges::range_value_t< To > > >
{
    using VertexType = std::common_type_t< std::remove_cvref_t< std::ranges::range_value_t< From > >,
                                           std::remove_cvref_t< std::ranges::range_value_t< To > > >;
    if (std::ranges::empty(from))
        return CrsGraph< VertexType >{};
    auto vert_map = robin_hood::unordered_flat_map< VertexType, robin_hood::unordered_flat_set< VertexType > >{};
    auto max_vert = std::numeric_limits< VertexType >::min();
    for (const auto& [f, t] : std::views::zip(std::forward< From >(from), std::forward< To >(to)))
    {
        const auto v_from = static_cast< VertexType >(f);
        const auto v_to   = static_cast< VertexType >(t);
        max_vert          = std::max(max_vert, v_from);
        max_vert          = std::max(max_vert, v_to);
        vert_map[v_from].insert(v_to);
        if (graph_type == GraphType::Undirected)
            vert_map[v_to].insert(v_from);
    }
    auto degrees = std::views::iota(VertexType{0}, max_vert + 1) | std::views::transform([&](auto v) {
                       const auto iter = vert_map.find(v);
                       return iter == vert_map.end() ? size_t{0} : iter->second.size();
                   });
    auto retval  = CrsGraph< VertexType >{degrees};
    for (const auto& [v, dests] : vert_map)
    {
        const auto row = retval(v);
        std::ranges::copy(dests, row.begin());
        std::ranges::sort(row);
    }
    return retval;
}
} // namespace lstr::util
#endif // L3STER_UTIL_ALGORITHM_HPP
