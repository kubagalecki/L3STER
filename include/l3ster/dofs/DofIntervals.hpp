#ifndef L3STER_DOFS_DOFINTERVALS_HPP
#define L3STER_DOFS_DOFINTERVALS_HPP

#include "l3ster/dofs/NodeCondensation.hpp"
#include "l3ster/util/Caliper.hpp"

namespace lstr::dofs
{
namespace detail
{
template < size_t n_fields >
using node_interval_t = std::pair< std::array< n_id_t, 2 >, std::bitset< n_fields > >;
template < size_t n_fields >
using node_interval_vector_t = std::vector< node_interval_t< n_fields > >;

template < CondensationPolicy CP, size_t n_fields >
auto computeDofIntervalsFromNodeData(const NodeCondensationMap< CP >&              cond_map,
                                     const std::vector< std::bitset< n_fields > >& field_cov)
    -> node_interval_vector_t< n_fields >
{
    const auto& nodes   = cond_map.getCondensedIds();
    auto        retval  = node_interval_vector_t< n_fields >{};
    auto        node_it = begin(nodes);
    while (node_it != end(nodes))
    {
        auto cov_it = std::next(begin(field_cov), std::distance(begin(nodes), node_it));
        node_it     = std::ranges::mismatch(std::ranges::subrange(node_it, end(nodes)), std::views::iota(*node_it)).in1;
        const auto adj_cov_end = std::next(begin(field_cov), std::distance(begin(nodes), node_it));
        while (cov_it != adj_cov_end)
        {
            const auto cov_pos_begin = std::distance(begin(field_cov), cov_it);
            const auto current_cov   = *cov_it;
            cov_it                 = std::find_if(cov_it, adj_cov_end, [&](const auto& c) { return c != current_cov; });
            const auto cov_pos_end = std::distance(begin(field_cov), cov_it);
            retval.emplace_back(std::array{nodes[cov_pos_begin], nodes[cov_pos_end - 1]}, current_cov);
        }
    }
    retval.shrink_to_fit();
    return retval;
}

template < CondensationPolicy CP, ProblemDef_c auto problem_def, el_o_t... orders >
auto makeFieldCoverageVector(const mesh::MeshPartition< orders... >& mesh,
                             const NodeCondensationMap< CP >&        cond_map,
                             util::ConstexprValue< problem_def >)
    -> std::vector< std::bitset< lstr::detail::deduceNFields(problem_def) > >
{
    auto retval =
        std::vector< std::bitset< lstr::detail::deduceNFields(problem_def) > >(cond_map.getCondensedIds().size());
    for (const auto& [dom_id, field_array] : problem_def)
    {
        const auto fields = util::toBitset(field_array);
        mesh.visit(
            [&](const auto& element) {
                for (auto node : getPrimaryNodesView< CP >(element))
                {
                    const auto local_node_id = cond_map.getLocalCondensedId(node);
                    retval[local_node_id] |= fields;
                }
            },
            dom_id);
    }
    return retval;
}

template < CondensationPolicy CP, ProblemDef_c auto problem_def, el_o_t... orders >
auto computeLocalDofIntervals(const mesh::MeshPartition< orders... >& mesh,
                              const NodeCondensationMap< CP >&        cond_map,
                              util::ConstexprValue< problem_def >     problemdef_ctwrapper)
    -> node_interval_vector_t< lstr::detail::deduceNFields(problem_def) >
{
    const auto field_coverage = makeFieldCoverageVector(mesh, cond_map, problemdef_ctwrapper);
    return computeDofIntervalsFromNodeData(cond_map, field_coverage);
}

template < size_t n_fields >
void serializeDofIntervals(const node_interval_vector_t< n_fields >&       intervals,
                           std::output_iterator< unsigned long long > auto out_it)
{
    for (const auto& [delims, field_cov] : intervals)
    {
        out_it = std::ranges::copy(delims, out_it).out;
        out_it = std::ranges::copy(util::serializeBitset(field_cov), out_it).out;
    }
}

template < size_t n_fields >
void deserializeDofIntervals(const std::ranges::sized_range auto& serial_data, auto out_it)
    requires std::same_as< std::ranges::range_value_t< std::decay_t< decltype(serial_data) > >, unsigned long long > and
             std::output_iterator< decltype(out_it), node_interval_t< n_fields > >
{
    constexpr auto n_ulls = util::bitsetNUllongs< n_fields >();
    for (auto data_it = std::ranges::begin(serial_data); data_it != std::ranges::end(serial_data);)
    {
        auto delims          = std::array< n_id_t, 2 >{};
        auto serial_fieldcov = std::array< unsigned long long, n_ulls >{};
        data_it              = std::ranges::copy_n(data_it, delims.size(), delims.begin()).in;
        data_it              = std::ranges::copy_n(data_it, n_ulls, serial_fieldcov.begin()).in;
        *out_it++ = std::make_pair(delims, util::trimBitset< n_fields >(util::deserializeBitset(serial_fieldcov)));
    }
}

template < size_t n_fields >
auto gatherGlobalDofIntervals(const MpiComm& comm, const node_interval_vector_t< n_fields >& local_intervals)
    -> node_interval_vector_t< n_fields >
{
    constexpr size_t serial_interval_size = util::bitsetNUllongs< n_fields >() + 2;
    const size_t     n_intervals_local    = local_intervals.size();
    const auto       comm_size            = comm.getSize();
    const auto       my_rank              = comm.getRank();

    const size_t max_n_intervals_global = std::invoke([&] {
        size_t retval{};
        comm.allReduce(std::views::single(n_intervals_local), &retval, MPI_MAX);
        return retval;
    });
    const size_t max_msg_size           = max_n_intervals_global * serial_interval_size + 1u;
    auto         serial_local_intervals = std::invoke([&] {
        auto retval    = util::ArrayOwner< unsigned long long >{max_msg_size};
        retval.front() = n_intervals_local;
        serializeDofIntervals(local_intervals, std::next(retval.begin()));
        return retval;
    });

    node_interval_vector_t< n_fields > retval;
    retval.reserve(comm_size * max_n_intervals_global);
    auto       proc_buf     = util::ArrayOwner< unsigned long long >{max_msg_size};
    const auto process_data = [&](int sender_rank) {
        if (sender_rank != my_rank)
        {
            const auto received_intervals =
                proc_buf | std::views::drop(1) | std::views::take(proc_buf.front() * serial_interval_size);
            deserializeDofIntervals< n_fields >(received_intervals, std::back_inserter(retval));
        }
        else
            std::ranges::copy(local_intervals, std::back_inserter(retval));
    };

    auto msg_buf =
        my_rank == 0 ? std::move(serial_local_intervals) : util::ArrayOwner< unsigned long long >{max_msg_size};
    auto request = comm.broadcastAsync(msg_buf, 0);
    for (int root_rank = 1; root_rank < comm_size; ++root_rank)
    {
        request.wait();
        std::swap(msg_buf, proc_buf);
        if (my_rank == root_rank)
            msg_buf = std::move(serial_local_intervals);
        request = comm.broadcastAsync(msg_buf, root_rank);
        process_data(root_rank - 1);
    }
    request.wait();
    std::swap(msg_buf, proc_buf);
    process_data(comm_size - 1);
    return retval;
}

template < size_t n_fields >
void consolidateDofIntervals(node_interval_vector_t< n_fields >& intervals)
{
    using interval_t = node_interval_t< n_fields >;

    const auto sort = [&intervals]< typename F >(F&& proj) {
        std::ranges::sort(intervals, std::less<>{}, std::forward< F >(proj));
    };
    constexpr auto by_dof = [](const interval_t& interval) {
        return std::make_pair(util::serializeBitset(interval.second), interval.first);
    };
    constexpr auto by_delim = [](const interval_t& interval) {
        return std::make_pair(interval.first, util::serializeBitset(interval.second));
    };
    const auto consolidate_samedof_overlappingdelim = [&intervals] {
        constexpr auto same_dof_overlapping = [](const auto& i1, const auto& i2) {
            const auto [lo1, hi1] = i1.first;
            const auto [lo2, hi2] = i2.first;
            return hi1 >= lo2 - 1u and i1.second == i2.second;
        };
        constexpr auto consolidate = [](const auto& i1, const auto& i2) {
            const auto [lo1, hi1] = i1.first;
            const auto [lo2, hi2] = i2.first;
            return std::make_pair(std::array{lo1, std::max(hi1, hi2)}, i1.second);
        };
        const auto erase_range = util::reduceConsecutive(intervals, same_dof_overlapping, consolidate);
        intervals.erase(erase_range.begin(), erase_range.end());
    };
    const auto consolidate_same_delim = [&intervals] {
        constexpr auto same_interval_delim = [](const auto& i1, const auto& i2) {
            return i1.first == i2.first;
        };
        constexpr auto consolidate = [](const auto& i1, const auto& i2) {
            return std::make_pair(i1.first, i1.second | i2.second);
        };
        intervals.erase(begin(util::reduceConsecutive(intervals, same_interval_delim, consolidate)), intervals.end());
    };
    const auto resolve_overlapping = [&intervals] {
        // TODO: Optimize this function so that `overlap_range` only covers the intervals which *end* before it
        // does. This will help reduce the n in the O(n^2) part of the algorithm

        std::vector< interval_t > scratchpad;
        auto                      overlap_begin = intervals.begin();
        while (overlap_begin != intervals.end())
        {
            auto       overlap_max_node = overlap_begin->first[1];
            const auto overlap_end      = std::ranges::find_if(
                std::next(overlap_begin),
                intervals.end(),
                [&overlap_max_node](const auto& delim) {
                    const auto [lo, hi]        = delim;
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
                const auto                       hi = std::invoke([&] {
                    auto retval = std::numeric_limits< n_id_t >::max();
                    for (const auto [int_lo, int_hi] : overlap_range | std::views::keys)
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
                });
                typename interval_t::second_type field_cov;
                for (const auto& [delims, cov] : overlap_range)
                {
                    const auto [int_lo, int_hi] = delims;
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
    intervals.shrink_to_fit();
}

template < size_t n_fields >
auto computeIntervalStarts(const node_interval_vector_t< n_fields >& intervals) -> std::vector< size_t >
{
    std::vector< size_t > retval(intervals.size());
    std::transform_exclusive_scan(
        begin(intervals), end(intervals), begin(retval), 0, std::plus< size_t >{}, [](const auto& interval) -> size_t {
            const auto& [delim, cov] = interval;
            const auto [lo, hi]      = delim;
            return (hi - lo + 1u) * cov.count();
        });
    return retval;
}

template < std::input_iterator I, std::sentinel_for< I > S >
I findNodeInterval(I begin, S end, n_id_t node)
{
    return std::ranges::lower_bound(begin, end, node, {}, [](const auto& interval) { return interval.first.back(); });
}
} // namespace detail

template < CondensationPolicy CP, ProblemDef_c auto problem_def, el_o_t... orders >
auto computeDofIntervals(const MpiComm&                          comm,
                         const mesh::MeshPartition< orders... >& mesh,
                         const NodeCondensationMap< CP >&        cond_map,
                         util::ConstexprValue< problem_def >     problemdef_ctwrapper)
    -> detail::node_interval_vector_t< lstr::detail::deduceNFields(problem_def) >
{
    L3STER_PROFILE_FUNCTION;
    const auto local_intervals = detail::computeLocalDofIntervals(mesh, cond_map, problemdef_ctwrapper);
    auto       retval          = detail::gatherGlobalDofIntervals(comm, local_intervals);
    detail::consolidateDofIntervals(retval);
    return retval;
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_DOFINTERVALS_HPP
