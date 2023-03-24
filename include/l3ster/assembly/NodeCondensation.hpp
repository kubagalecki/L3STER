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
constexpr decltype(auto) getPrimaryNodes(const Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
        return getBoundaryNodes(element);
}
template < CondensationPolicy CP, ElementTypes ET, el_o_t EO >
consteval auto getNumPrimaryNodes(ValuePack< CP, ET, EO > = {}) -> size_t
{
    return std::tuple_size_v< std::decay_t< decltype(getPrimaryNodes< CP >(std::declval< Element< ET, EO > >())) > >;
};

template < CondensationPolicy CP, ProblemDef_c auto problem_def >
auto getActiveNodes(const MeshPartition& mesh, ConstexprValue< problem_def >, CondensationPolicyTag< CP > = {})
    -> std::vector< n_id_t >
{
    robin_hood::unordered_flat_set< n_id_t > active_nodes_set;
    mesh.visit(
        [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            for (auto n : getPrimaryNodes< CP >(element))
                active_nodes_set.insert(n);
        },
        problem_def | std::views::transform([](const auto& pair) { return pair.first; }));
    std::vector< n_id_t > retval;
    retval.reserve(active_nodes_set.size());
    std::ranges::copy(active_nodes_set, std::back_inserter(retval));
    std::ranges::sort(retval);
    return retval;
}

inline auto computeCondensedElementBoundaryNodeIds(const MpiComm&               comm,
                                                   const MeshPartition&         mesh,
                                                   const std::vector< n_id_t >& uncondensed_boundary_nodes)
    -> std::vector< n_id_t >
{
    const int my_rank   = comm.getRank();
    const int comm_size = comm.getSize();

    const size_t n_owned_boundary_nodes =
        std::ranges::count_if(uncondensed_boundary_nodes, [&](auto node) { return mesh.isOwnedNode(node); });
    const auto max_msg_size             = std::invoke([&] {
        size_t retval{};
        comm.allReduce(std::views::single(n_owned_boundary_nodes + 1u), &retval, MPI_MAX);
        return retval;
    });
    auto       owned_boundary_nodes_msg = std::invoke([&] {
        ArrayOwner< n_id_t > retval(max_msg_size);
        retval.front() = n_owned_boundary_nodes;
        std::ranges::copy_if(
            uncondensed_boundary_nodes, std::next(retval.begin()), [&](auto node) { return mesh.isOwnedNode(node); });
        return retval;
    });

    std::vector< n_id_t > retval(uncondensed_boundary_nodes.size());
    if (comm_size == 1)
    {
        std::iota(retval.begin(), retval.end(), 0ul);
        return retval;
    }
    ArrayOwner< n_id_t > proc_buf(max_msg_size), comm_buf(max_msg_size);
    const auto           process_received_boundary_ids = [&](int sender_rank) {
        auto&      buf_to_process = sender_rank == my_rank ? owned_boundary_nodes_msg : proc_buf;
        const auto off_rank_nodes = buf_to_process | std::views::drop(1) | std::views::take(buf_to_process.front());
        for (size_t i = 0; auto& cond_id : retval)
            cond_id += std::distance(off_rank_nodes.begin(),
                                     std::ranges::lower_bound(off_rank_nodes, uncondensed_boundary_nodes[i++]));
    };

    auto request = comm.broadcastAsync(my_rank == 0 ? owned_boundary_nodes_msg : comm_buf, 0);
    for (int sender_rank = 1; sender_rank < comm_size; ++sender_rank)
    {
        request.wait();
        std::swap(proc_buf, comm_buf);
        request = comm.broadcastAsync(my_rank == sender_rank ? owned_boundary_nodes_msg : comm_buf, sender_rank);
        process_received_boundary_ids(sender_rank - 1);
    }
    request.wait();
    std::swap(proc_buf, comm_buf);
    process_received_boundary_ids(comm_size - 1);
    return retval;
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

template < CondensationPolicy CP, ProblemDef_c auto problem_def >
auto makeCondensationMap(const MpiComm&                comm,
                         const MeshPartition&          mesh,
                         ConstexprValue< problem_def > probdef_ctwrpr,
                         CondensationPolicyTag< CP >   cp_tag = {}) -> NodeCondensationMap< CP >
{
    const auto boundary_nodes           = getActiveNodes(mesh, probdef_ctwrpr, cp_tag);
    auto       condensed_boundary_nodes = computeCondensedElementBoundaryNodeIds(comm, mesh, boundary_nodes);
    return {boundary_nodes, condensed_boundary_nodes};
}

template < CondensationPolicy CP >
inline auto getLocalCondensedId(const NodeCondensationMap< CP >& cond_map, n_id_t node) -> n_id_t
{
    return static_cast< n_id_t >(
        std::distance(cond_map.getCondensedIds().begin(),
                      std::ranges::lower_bound(cond_map.getCondensedIds(), cond_map.getCondensedId(node))));
}

template < CondensationPolicy CP >
auto getCondensedOwnedNodesView(const MeshPartition& mesh, const NodeCondensationMap< CP >& cond_map)
{
    return cond_map.getCondensedIds() |
           std::views::filter([&](n_id_t node) { return mesh.isOwnedNode(cond_map.getUncondensedId(node)); });
}
} // namespace detail
} // namespace lstr
#endif // L3STER_NODECONDENSATION_HPP
