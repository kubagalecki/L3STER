#ifndef L3STER_ASSEMBLY_DOFINTERVALS_HPP
#define L3STER_ASSEMBLY_DOFINTERVALS_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/BitsetManip.hpp"

namespace lstr
{
namespace detail
{
constexpr size_t getNFields(const auto& problem_def)
{
    // problem_def_t == array< pair< d_id_t, array< bool, fields > >, n_domains >
    return std::tuple_size_v< typename std::decay_t< decltype(problem_def) >::value_type::second_type >;
}

constexpr size_t getFieldUllongSize(const auto& problem_def)
{
    using problem_def_t = std::decay_t< decltype(problem_def) >;
    return bitsetNUllongs< getNFields(problem_def_t{}) >();
}

constexpr size_t getSerialDofIntervalSize(const auto& problem_def)
{
    return getFieldUllongSize(problem_def) + 2u;
}

template < size_t n_fields >
using node_interval_t = std::pair< std::array< n_id_t, 2 >, std::bitset< n_fields > >;
template < size_t n_fields >
using node_interval_vector_t = std::vector< node_interval_t< n_fields > >;

template < size_t n_fields >
auto computeDofIntervalsFromNodeData(const std::vector< n_id_t >&                  nodes,
                                     const std::vector< std::bitset< n_fields > >& field_cov,
                                     ConstexprValue_c auto                         problemdef_ctwrapper)
{
    node_interval_vector_t< getNFields(decltype(problemdef_ctwrapper)::value) > retval;
    constexpr size_t                                                            reserve_heuristic = 16;
    retval.reserve(nodes.size() / reserve_heuristic);
    for (ptrdiff_t i = 0; i < static_cast< ptrdiff_t >(nodes.size() - 1); ++i)
    {
        std::array< n_id_t, 2 > int_delim{};
        auto& [int_first, int_last] = int_delim;
        int_first                   = nodes[i];
        auto int_cov                = field_cov[i];
        while (i < static_cast< ptrdiff_t >(nodes.size() - 1) and nodes[i] + 1 == nodes[i + 1] and
               field_cov[i + 1] == int_cov)
            ++i;
        int_last = nodes[i];
        retval.emplace_back(int_delim, int_cov);
    }
    retval.shrink_to_fit();
    return retval;
}

auto computeLocalDofIntervals(const MeshPartition& mesh, ConstexprValue_c auto problemdef_ctwrapper)
{
    constexpr auto& problem_def = decltype(problemdef_ctwrapper)::value;
    constexpr auto  n_domains   = problem_def.size();
    constexpr auto  n_fields    = getNFields(problem_def);
    using field_coverage_t      = std::bitset< n_fields >;

    std::array< std::pair< d_id_t, std::bitset< n_fields > >, n_domains > problem_def_converted;
    std::ranges::transform(
        problem_def, begin(problem_def_converted), [](const Pair< d_id_t, std::array< bool, n_fields > >& pair_ce) {
            return std::make_pair(pair_ce.first, toBitset(pair_ce.second));
        });

    const auto            n_nodes = mesh.getNodes().size() + mesh.getGhostNodes().size();
    std::vector< n_id_t > node_ids;
    node_ids.reserve(n_nodes);
    std::ranges::merge(mesh.getNodes(), mesh.getGhostNodes(), std::back_inserter(node_ids));

    std::vector< field_coverage_t > field_coverage(n_nodes);
    for (const auto& [dom_id, fields] : problem_def_converted)
    {
        mesh.cvisit(
            [&](const auto& element) {
                for (auto node : element.getNodes())
                {
                    const auto local_node_id = std::distance(begin(node_ids), std::ranges::lower_bound(node_ids, node));
                    field_coverage[local_node_id] |= fields;
                }
            },
            {dom_id});
    }

    return computeDofIntervalsFromNodeData(std::move(node_ids), std::move(field_coverage), problemdef_ctwrapper);
}

template < size_t n_fields >
void serializeDofIntervals(const node_interval_vector_t< n_fields >&       intervals,
                           std::output_iterator< unsigned long long > auto out_it)
{
    for (const auto& [delims, field_cov] : intervals)
    {
        *out_it++ = delims[0];
        *out_it++ = delims[1];
        for (auto chunk : serializeBitset(field_cov))
            *out_it++ = chunk;
    }
}

template < size_t n_fields >
    auto deserializeDofIntervals(const std::ranges::sized_range auto&serial_data, auto out_it) requires std::same_as <
    std::ranges::range_value_t< std::decay_t< decltype(serial_data) > >,
unsigned long long >
    and std::output_iterator< decltype(out_it), std::pair< std::array< n_id_t, 2 >, std::bitset< n_fields > > >
{
    for (auto data_it = std::ranges::begin(serial_data); data_it != std::ranges::end(serial_data);)
    {
        std::array< n_id_t, 2 > delims{};
        delims[0] = *data_it++;
        delims[1] = *data_it++;
        std::array< unsigned long long, bitsetNUllongs< n_fields >() > serial_fieldcov;
        for (ptrdiff_t i = 0; i < static_cast< ptrdiff_t >(bitsetNUllongs< n_fields >()); ++i)
            serial_fieldcov[i] = *data_it++;
        *out_it++ = std::make_pair(delims, trimBitset< n_fields >(deserializeBitset(serial_fieldcov)));
    }
}

auto gatherGlobalDofIntervals(const auto&           local_intervals,
                              ConstexprValue_c auto problemdef_ctwrapper,
                              const MpiComm&        comm)
{
    constexpr auto& problem_def          = decltype(problemdef_ctwrapper)::value;
    constexpr auto  serial_interval_size = getSerialDofIntervalSize(problem_def);
    constexpr int   reduce_size = 1, reduce_root_rank = 0;
    const size_t    n_intervals_local = local_intervals.size();
    const auto      comm_size         = comm.getSize();
    const auto      my_rank           = comm.getRank();

    size_t max_n_intervals_global{};
    comm.reduce(&n_intervals_local, &max_n_intervals_global, reduce_size, reduce_root_rank, MPI_MAX);
    comm.broadcast(&max_n_intervals_global, reduce_size, reduce_root_rank);
    const auto max_msg_size = max_n_intervals_global * serial_interval_size + 1;

    auto serial_local_intervals = std::make_unique_for_overwrite< unsigned long long[] >(max_msg_size); // NOLINT
    serial_local_intervals[0]   = n_intervals_local;
    serializeDofIntervals(local_intervals, std::next(serial_local_intervals.get()));
    const auto local_msg_size = n_intervals_local * serial_interval_size + 1;

    node_interval_vector_t< getNFields(problem_def) > intervals;
    std::vector< ptrdiff_t >                          interval_inds;
    intervals.reserve(comm_size * max_n_intervals_global);
    interval_inds.reserve(comm_size + 1);
    interval_inds.push_back(0);

    auto       proc_buf              = std::make_unique_for_overwrite< unsigned long long[] >(max_msg_size); // NOLINT
    const auto process_received_data = [&]() {                                                               // NOLINT
        const size_t n_int_rcvd = proc_buf[0];
        deserializeDofIntervals< getNFields(problem_def) >(
            std::views::counted(std::next(proc_buf.get()), static_cast< ptrdiff_t >(n_int_rcvd * serial_interval_size)),
            std::back_inserter(intervals));
        interval_inds.push_back(n_int_rcvd);
    };
    const auto process_my_data = [&] {
        std::ranges::copy(local_intervals, std::back_inserter(intervals));
        interval_inds.push_back(n_intervals_local);
    };

    auto msg_buf = my_rank == 0 ? std::move(serial_local_intervals)
                                : std::make_unique_for_overwrite< unsigned long long[] >(max_msg_size); // NOLINT
    auto request = comm.broadcastAsync(msg_buf.get(), my_rank == 0 ? local_msg_size : max_msg_size, 0);
    for (int root_rank = 1; root_rank < comm_size; ++root_rank)
    {
        request.wait();
        std::swap(msg_buf, proc_buf);
        if (my_rank == root_rank)
            msg_buf = std::move(serial_local_intervals);
        request = comm.broadcastAsync(msg_buf.get(), my_rank == root_rank ? local_msg_size : max_msg_size, root_rank);
        my_rank != root_rank - 1 ? process_received_data() : process_my_data();
    }
    request.wait();
    proc_buf = std::move(msg_buf);
    my_rank == comm_size - 1 ? process_my_data() : process_received_data();

    std::inclusive_scan(begin(interval_inds), end(interval_inds), begin(interval_inds));
    return std::make_pair(std::move(interval_inds), std::move(intervals));
}

template < size_t n_fields >
void consolidateDofIntervals(node_interval_vector_t< n_fields >& intervals)
{
    using interval_t = node_interval_t< n_fields >;

    const auto sort = [&intervals]< typename F >(F&& proj) {
        std::ranges::sort(intervals, std::less<>{}, std::forward< F >(proj));
    };
    constexpr auto by_dof = [](const interval_t& interval) {
        return std::make_pair(serializeBitset(interval.second), interval.first);
    };
    constexpr auto by_delim = [](const interval_t& interval) {
        return std::make_pair(interval.first, serializeBitset(interval.second));
    };

    const auto consolidate_samedof_overlappingdelim = [&intervals] {
        constexpr auto same_dof_overlapping = [](const auto& i1, const auto& i2) {
            const auto& [lo1, hi1] = i1.first;
            const auto& [lo2, hi2] = i2.first;
            return hi1 >= lo2 - 1u and i1.second == i2.second;
        };
        constexpr auto consolidate = [](const auto& i1, const auto& i2) {
            const auto& [lo1, hi1] = i1.first;
            const auto& [lo2, hi2] = i2.first;
            return std::make_pair(std::array{lo1, std::max(hi1, hi2)}, i1.second);
        };
        intervals.erase(begin(reduceConsecutive(intervals, same_dof_overlapping, consolidate)), intervals.end());
    };
    const auto consolidate_same_delim = [&intervals] {
        constexpr auto same_interval_delim = [](const auto& i1, const auto& i2) {
            return i1.first == i2.first;
        };
        constexpr auto consolidate = [](const auto& i1, const auto& i2) {
            return std::make_pair(i1.first, i1.second | i2.second);
        };
        intervals.erase(begin(reduceConsecutive(intervals, same_interval_delim, consolidate)), intervals.end());
    };

    const auto resolve_overlapping = [&intervals] {
        // TODO: Optimize this function so that `overlap_range` only covers the intervals which *end* before it does.
        // This will help reduce the n int the O(n^2) part of the algorithm

        std::vector< interval_t > scratchpad;
        auto                      overlap_begin = intervals.begin();
        while (overlap_begin != intervals.end())
        {
            auto       overlap_max_node = overlap_begin->first[1];
            const auto overlap_end      = std::ranges::find_if(
                std::next(overlap_begin),
                intervals.end(),
                [&overlap_max_node](const auto& delim) {
                    const auto& [lo, hi]       = delim;
                    const bool outside_overlap = lo > overlap_max_node;
                    if (not outside_overlap)
                        overlap_max_node = std::max(overlap_max_node, hi);
                    return outside_overlap;
                },
                [](const interval_t& interval) -> const auto& { return interval.first; });
            const auto overlap_range = std::ranges::subrange(overlap_begin, overlap_end);
            const auto overlap_size  = overlap_range.size();
            if (overlap_size == 1)
            {
                std::advance(overlap_begin, 1);
                continue;
            }

            auto lo = overlap_begin->first[0];
            scratchpad.clear();
            scratchpad.reserve(2 * overlap_size - 1);
            while (lo <= overlap_max_node)
            {
                const auto hi = [&] {
                    auto retval = std::numeric_limits< n_id_t >::max();
                    for (const auto& [int_lo, int_hi] : overlap_range | std::views::keys)
                    {
                        if (int_lo == int_hi and int_lo == lo)
                            return lo;
                        if (int_lo > lo)
                            retval = std::min(retval, int_lo - 1);
                        if (int_hi >= lo)
                            retval = std::min(retval, int_hi);
                        if (int_lo + 1 > retval)
                            break;
                    }
                    return retval;
                }();
                typename interval_t::second_type field_cov;
                for (const auto& [delims, cov] : overlap_range)
                {
                    const auto& [int_lo, int_hi] = delims;
                    if (int_lo <= lo and int_hi >= hi)
                        field_cov |= cov;
                    if (int_lo > hi)
                        break;
                }
                scratchpad.emplace_back(std::array{lo, hi}, field_cov);
                lo = hi + 1;
            }
            if (scratchpad.size() <= overlap_size)
            {
                std::ranges::copy(scratchpad, overlap_begin);
                overlap_begin = intervals.erase(overlap_begin + scratchpad.size(), overlap_end);
            }
            else
            {
                std::copy_n(scratchpad.begin(), overlap_size, overlap_begin);
                overlap_begin = intervals.insert(overlap_end, scratchpad.begin() + overlap_size, scratchpad.end());
                std::advance(overlap_begin, scratchpad.size() - overlap_size);
            }
        }
    };

    sort(by_dof);
    consolidate_samedof_overlappingdelim();
    sort(by_delim);
    consolidate_same_delim();
    resolve_overlapping();
    consolidate_samedof_overlappingdelim();
}

template < size_t n_fields >
auto computeIntervalStarts(const node_interval_vector_t< n_fields >& intervals)
{
    std::vector< global_dof_t > retval(intervals.size());
    std::transform_exclusive_scan(begin(intervals),
                                  end(intervals),
                                  begin(retval),
                                  0,
                                  std::plus< global_dof_t >{},
                                  [](const auto& interval) -> global_dof_t {
                                      const auto& [delim, cov] = interval;
                                      const auto& [lo, hi]     = delim;
                                      return (hi - lo + 1) * cov.count();
                                  });
    return retval;
}

template < std::input_iterator I, std::sentinel_for< I > S >
I findNodeInterval(I begin, S end, n_id_t node)
{
    return std::ranges::find_if(
        begin,
        end,
        [node = node](n_id_t hi) { return hi >= node; },
        [](const auto& interval) { return interval.first[1]; });
}
} // namespace detail

auto computeDofIntervals(const MeshPartition& mesh, ConstexprValue_c auto problemdef_ctwrapper, const MpiComm& comm)
{
    const auto local_intervals  = detail::computeLocalDofIntervals(mesh, problemdef_ctwrapper);
    auto       global_data      = detail::gatherGlobalDofIntervals(local_intervals, problemdef_ctwrapper, comm);
    auto& [_, global_intervals] = global_data;
    detail::consolidateDofIntervals(global_intervals);
    return std::move(global_intervals);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_DOFINTERVALS_HPP
