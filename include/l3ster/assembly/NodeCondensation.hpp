#ifndef L3STER_NODECONDENSATION_HPP
#define L3STER_NODECONDENSATION_HPP

#include "l3ster/assembly/ProblemDefinition.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

namespace lstr
{
enum struct CondensationPolicy
{
    None,
    ElementBoundary
};

template < CondensationPolicy cond_policy >
struct CondensationPolicyTag
{
    static constexpr auto value = cond_policy;
};

inline constexpr auto no_condensation  = CondensationPolicyTag< CondensationPolicy::None >{};
inline constexpr auto element_boundary = CondensationPolicyTag< CondensationPolicy::ElementBoundary >{};

namespace detail
{
template < typename T >
struct IsCondensationPolicyTag : std::false_type
{};
template < CondensationPolicy CP >
struct IsCondensationPolicyTag< CondensationPolicyTag< CP > > : std::true_type
{};
template < typename T >
concept CondensationPolicyTag_c = IsCondensationPolicyTag< T >::value;

template < CondensationPolicy CP, ElementTypes ET, el_o_t EO >
constexpr decltype(auto) getPrimaryNodesView(const Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
        return getBoundaryNodes(element);
}
template < CondensationPolicy CP, ElementTypes ET, el_o_t EO >
constexpr decltype(auto) getPrimaryNodesArray(const Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
    {
        std::array< n_id_t, ElementTraits< Element< ET, EO > >::boundary_node_inds.size() > retval;
        std::ranges::copy(getBoundaryNodes(element), retval.begin());
        return retval;
    }
}
template < CondensationPolicy CP, ElementTypes ET, el_o_t EO >
consteval auto getNumPrimaryNodes(ValuePack< CP, ET, EO > = {}) -> size_t
{
    return std::tuple_size_v<
        std::decay_t< decltype(getPrimaryNodesArray< CP >(std::declval< Element< ET, EO > >())) > >;
};

template < CondensationPolicy CP, ProblemDef_c auto problem_def, el_o_t... orders >
auto getActiveNodes(const MeshPartition< orders... >& mesh,
                    ConstexprValue< problem_def >,
                    CondensationPolicyTag< CP > = {}) -> std::vector< n_id_t >
{
    auto active_nodes_set = robin_hood::unordered_flat_set< n_id_t >{};
    mesh.visit(
        [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            for (auto n : getPrimaryNodesView< CP >(element))
                active_nodes_set.insert(n);
        },
        problem_def | std::views::transform([](const auto& pair) { return pair.first; }));
    auto retval = std::vector< n_id_t >{};
    retval.reserve(active_nodes_set.size());
    std::ranges::copy(active_nodes_set, std::back_inserter(retval));
    std::ranges::sort(retval);
    return retval;
}

auto packRangeWithSizeForComm(std::ranges::range auto&& range, size_t alloc_size)
    -> ArrayOwner< std::ranges::range_value_t< decltype(range) > >
    requires mpi::MpiType_c< std::ranges::range_value_t< decltype(range) > > and
             std::integral< std::ranges::range_value_t< decltype(range) > >
{
    using range_value_t             = std::ranges::range_value_t< decltype(range) >;
    const size_t range_size         = std::ranges::distance(range);
    const auto   range_size_to_pack = exactIntegerCast< range_value_t >(range_size);
    auto         retval             = ArrayOwner< range_value_t >(alloc_size);
    retval.front()                  = range_size_to_pack;
    std::ranges::copy(std::forward< decltype(range) >(range), std::next(retval.begin()));
    return retval;
}

template < std::integral T >
auto unpackRangeFromComm(ArrayOwner< T >& message)
{
    return message | std::views::drop(1) | std::views::take(static_cast< size_t >(message.front()));
}

template < el_o_t... orders >
auto computeCondensedActiveNodeIds(const MpiComm&                    comm,
                                   const MeshPartition< orders... >& mesh,
                                   const std::vector< n_id_t >&      uncondensed_active_nodes) -> std::vector< n_id_t >
{
    const int my_rank   = comm.getRank();
    const int comm_size = comm.getSize();

    // The condensation algorithm requires that each uncondensed active node be owned by exactly one partition. A node
    // may be inactive on the partition which owns it. Such nodes must therefore first be claimed by a different
    // partition.
    const auto owned_active_nodes     = std::invoke([&] {
        auto retval = std::vector< n_id_t >{};
        retval.reserve(uncondensed_active_nodes.size());
        for (auto n : uncondensed_active_nodes)
            if (mesh.isOwnedNode(n))
                retval.push_back(n);
        retval.shrink_to_fit();
        return retval;
    });
    const auto max_msg_size           = std::invoke([&] {
        size_t retval{};
        comm.allReduce(std::views::single(owned_active_nodes.size() + 1u), &retval, MPI_MAX);
        return retval;
    });
    auto       owned_active_nodes_msg = packRangeWithSizeForComm(owned_active_nodes, max_msg_size);

    std::vector< n_id_t > retval(uncondensed_active_nodes.size());
    if (comm_size == 1)
    {
        std::iota(retval.begin(), retval.end(), 0ul);
        return retval;
    }
    ArrayOwner< n_id_t > proc_buf(max_msg_size), comm_buf(max_msg_size);
    const auto           process_received_active_ids = [&](int sender_rank) {
        const auto off_rank_nodes = unpackRangeFromComm(sender_rank == my_rank ? owned_active_nodes_msg : proc_buf);
        for (size_t i = 0; auto& cond_id : retval)
            cond_id += std::distance(off_rank_nodes.begin(),
                                     std::ranges::lower_bound(off_rank_nodes, uncondensed_active_nodes[i++]));
    };

    auto request = comm.broadcastAsync(my_rank == 0 ? owned_active_nodes_msg : comm_buf, 0);
    for (int sender_rank = 1; sender_rank < comm_size; ++sender_rank)
    {
        request.wait();
        std::swap(proc_buf, comm_buf);
        request = comm.broadcastAsync(my_rank == sender_rank ? owned_active_nodes_msg : comm_buf, sender_rank);
        process_received_active_ids(sender_rank - 1);
    }
    request.wait();
    std::swap(proc_buf, comm_buf);
    process_received_active_ids(comm_size - 1);
    return retval;
}

template < el_o_t... orders >
void activateOwned(const MpiComm& comm, const MeshPartition< orders... >& mesh, std::vector< n_id_t >& my_active_nodes)
{
    const int my_rank     = comm.getRank();
    const int comm_size   = comm.getSize();
    auto      ghost_nodes = std::vector< n_id_t >{};
    ghost_nodes.reserve(my_active_nodes.size());
    for (auto n : my_active_nodes)
        if (mesh.isGhostNode(n))
            ghost_nodes.push_back(n);
    const auto max_ghost_nodes              = std::invoke([&] {
        size_t retval{};
        comm.allReduce(std::views::single(ghost_nodes.size()), &retval, MPI_MAX);
        return retval;
    });
    const auto process_offrank_active_nodes = [&](ArrayOwner< n_id_t >& active_nodes) {
        for (auto active : unpackRangeFromComm(active_nodes))
            if (mesh.isOwnedNode(active) and not std::ranges::binary_search(my_active_nodes, active))
                my_active_nodes.insert(std::ranges::lower_bound(my_active_nodes, active), active);
    };
    auto my_active_ghost_nodes = packRangeWithSizeForComm(ghost_nodes, max_ghost_nodes + 1);
    auto comm_buf = ArrayOwner< n_id_t >(max_ghost_nodes + 1), proc_buf = ArrayOwner< n_id_t >(max_ghost_nodes + 1);
    auto request = comm.broadcastAsync(my_rank == 0 ? my_active_ghost_nodes : comm_buf, 0);
    for (int send_rank = 1; send_rank != comm_size; ++send_rank)
    {
        request.wait();
        std::swap(proc_buf, comm_buf);
        request = comm.broadcastAsync(my_rank == send_rank ? my_active_ghost_nodes : comm_buf, send_rank);
        if (my_rank != send_rank - 1)
            process_offrank_active_nodes(proc_buf);
    }
    request.wait();
    if (my_rank != comm_size - 1)
        process_offrank_active_nodes(comm_buf);
    my_active_nodes.shrink_to_fit();
}

template < CondensationPolicy >
class NodeCondensationMap
{
public:
    NodeCondensationMap(RangeOfConvertibleTo_c< n_id_t > auto&& nodes_to_condense, std::vector< n_id_t > condensed_ids)
        : m_condensed_ids{std::move(condensed_ids)}
    {
        m_forward_map.reserve(m_condensed_ids.size());
        m_inverse_map.reserve(m_condensed_ids.size());
        for (size_t i = 0; n_id_t uncond_node : nodes_to_condense)
        {
            const auto condensed_id = m_condensed_ids[i++];
            m_forward_map.emplace(uncond_node, condensed_id);
            m_inverse_map.emplace(condensed_id, uncond_node);
        }
    }

    [[nodiscard]] auto getCondensedId(n_id_t id) const -> n_id_t { return m_forward_map.at(id); }
    [[nodiscard]] auto getUncondensedId(n_id_t id) const -> n_id_t { return m_inverse_map.at(id); }
    [[nodiscard]] auto getCondensedIds() const -> std::span< const n_id_t > { return m_condensed_ids; }

private:
    std::vector< n_id_t >                            m_condensed_ids;
    robin_hood::unordered_flat_map< n_id_t, n_id_t > m_forward_map, m_inverse_map;
};

template < CondensationPolicy CP, ProblemDef_c auto problem_def, el_o_t... orders >
auto makeCondensationMap(const MpiComm&                    comm,
                         const MeshPartition< orders... >& mesh,
                         ConstexprValue< problem_def >     probdef_ctwrpr,
                         CondensationPolicyTag< CP >       cp_tag = {}) -> NodeCondensationMap< CP >
{
    auto active_nodes = getActiveNodes(mesh, probdef_ctwrpr, cp_tag);
    activateOwned(comm, mesh, active_nodes);
    auto condensed_active_nodes = computeCondensedActiveNodeIds(comm, mesh, active_nodes);
    return {std::move(active_nodes), std::move(condensed_active_nodes)};
}

template < CondensationPolicy CP >
inline auto getLocalCondensedId(const NodeCondensationMap< CP >& cond_map, n_id_t node) -> n_id_t
{
    return static_cast< n_id_t >(
        std::distance(cond_map.getCondensedIds().begin(),
                      std::ranges::lower_bound(cond_map.getCondensedIds(), cond_map.getCondensedId(node))));
}

template < CondensationPolicy CP, el_o_t... orders >
auto getCondensedOwnedNodesView(const MeshPartition< orders... >& mesh, const NodeCondensationMap< CP >& cond_map)
{
    return cond_map.getCondensedIds() |
           std::views::filter([&](n_id_t node) { return mesh.isOwnedNode(cond_map.getUncondensedId(node)); });
}

template < CondensationPolicy CP, el_o_t... orders >
auto getCondensedGhostNodesView(const MeshPartition< orders... >& mesh, const NodeCondensationMap< CP >& cond_map)
{
    return cond_map.getCondensedIds() |
           std::views::filter([&](n_id_t node) { return mesh.isGhostNode(cond_map.getUncondensedId(node)); });
}
} // namespace detail
} // namespace lstr
#endif // L3STER_NODECONDENSATION_HPP
