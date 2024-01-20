#ifndef L3STER_DOFS_NODECONDENSATION_HPP
#define L3STER_DOFS_NODECONDENSATION_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/dofs/ProblemDefinition.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/ArrayOwner.hpp"
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

inline constexpr auto no_condensation_tag  = CondensationPolicyTag< CondensationPolicy::None >{};
inline constexpr auto element_boundary_tag = CondensationPolicyTag< CondensationPolicy::ElementBoundary >{};

namespace dofs
{
template < typename T >
struct IsCondensationPolicyTag : std::false_type
{};
template < CondensationPolicy CP >
struct IsCondensationPolicyTag< CondensationPolicyTag< CP > > : std::true_type
{};
template < typename T >
concept CondensationPolicyTag_c = IsCondensationPolicyTag< T >::value;

template < CondensationPolicy CP, mesh::ElementType ET, el_o_t EO >
constexpr decltype(auto) getPrimaryNodesView(const mesh::Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
        return getBoundaryNodes(element);
}
template < CondensationPolicy CP, mesh::ElementType ET, el_o_t EO >
constexpr decltype(auto) getPrimaryNodesArray(const mesh::Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
    {
        std::array< n_id_t, mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds.size() > retval;
        std::ranges::copy(getBoundaryNodes(element), retval.begin());
        return retval;
    }
}
template < CondensationPolicy CP, mesh::ElementType ET, el_o_t EO >
consteval auto getNumPrimaryNodes(util::ValuePack< CP, ET, EO > = {}) -> size_t
{
    return std::tuple_size_v<
        std::decay_t< decltype(getPrimaryNodesArray< CP >(std::declval< mesh::Element< ET, EO > >())) > >;
};

template < CondensationPolicy CP, ProblemDef problem_def, el_o_t... orders >
auto getActiveNodes(const mesh::MeshPartition< orders... >& mesh,
                    util::ConstexprValue< problem_def >,
                    CondensationPolicyTag< CP > = {}) -> std::vector< n_id_t >
{
    auto active_nodes_set = robin_hood::unordered_flat_set< n_id_t >{};
    mesh.visit(
        [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            for (auto n : getPrimaryNodesView< CP >(element))
                active_nodes_set.insert(n);
        },
        problem_def | std::views::transform([](const DomainDef< problem_def.n_fields >& d) { return d.domain; }));
    auto retval = std::vector< n_id_t >{};
    retval.reserve(active_nodes_set.size());
    std::ranges::copy(active_nodes_set, std::back_inserter(retval));
    std::ranges::sort(retval);
    return retval;
}

template < el_o_t... orders >
auto computeCondensedActiveNodeIds(const MpiComm&                          comm,
                                   const mesh::MeshPartition< orders... >& mesh,
                                   const std::vector< n_id_t >& uncondensed_active_nodes) -> std::vector< n_id_t >
{
    // The condensation algorithm requires that each uncondensed active node be owned by exactly one partition. A node
    // may be inactive on the partition which owns it. Such nodes must therefore first be claimed by a different
    // partition.
    const auto owned_active_nodes = std::invoke([&] {
        auto retval = std::vector< n_id_t >{};
        retval.reserve(uncondensed_active_nodes.size());
        std::ranges::copy_if(uncondensed_active_nodes, std::back_inserter(retval), mesh.getOwnedNodePredicate());
        retval.shrink_to_fit();
        return retval;
    });

    std::vector< n_id_t > retval(uncondensed_active_nodes.size());
    if (comm.getSize() == 1)
    {
        std::iota(retval.begin(), retval.end(), 0ul);
        return retval;
    }
    const auto process_received_active_ids = [&](std::span< const n_id_t > off_rank_nodes, int) {
        for (size_t i = 0; i != retval.size(); ++i)
        {
            const auto iter = std::ranges::lower_bound(off_rank_nodes, uncondensed_active_nodes[i]);
            retval[i] += std::distance(off_rank_nodes.begin(), iter);
        }
    };
    util::staggeredAllGather(comm, std::span{owned_active_nodes}, process_received_active_ids);
    return retval;
}

template < el_o_t... orders >
void activateOwned(const MpiComm&                          comm,
                   const mesh::MeshPartition< orders... >& mesh,
                   std::vector< n_id_t >&                  my_active_nodes)
{
    const int  my_rank                      = comm.getRank();
    const auto my_active_ghost_nodes        = std::invoke([&] {
        auto retval = std::vector< n_id_t >{};
        retval.reserve(my_active_nodes.size());
        std::ranges::copy_if(my_active_nodes, std::back_inserter(retval), mesh.getGhostNodePredicate());
        return retval;
    });
    const auto process_offrank_active_nodes = [&](std::span< const n_id_t > active_nodes, int send_rank) {
        if (send_rank != my_rank)
            for (auto active : active_nodes)
                if (mesh.isOwnedNode(active) and not std::ranges::binary_search(my_active_nodes, active))
                    my_active_nodes.insert(std::ranges::lower_bound(my_active_nodes, active), active);
    };
    util::staggeredAllGather(comm, std::span{my_active_ghost_nodes}, process_offrank_active_nodes);
    my_active_nodes.shrink_to_fit();
}

template < CondensationPolicy >
class NodeCondensationMap
{
public:
    template < RangeOfConvertibleTo_c< n_id_t > NodeRange >
    NodeCondensationMap(NodeRange&& nodes_to_condense, std::vector< n_id_t > condensed_ids)
        : m_condensed_ids{std::move(condensed_ids)}
    {
        m_forward_map.reserve(m_condensed_ids.size());
        m_inverse_map.reserve(m_condensed_ids.size());
        for (size_t i = 0; n_id_t uncond_node : std::forward< NodeRange >(nodes_to_condense))
        {
            const auto condensed_id = m_condensed_ids[i++];
            m_forward_map.emplace(uncond_node, condensed_id);
            m_inverse_map.emplace(condensed_id, uncond_node);
        }
    }

    [[nodiscard]] auto getCondensedId(n_id_t id) const -> n_id_t { return m_forward_map.at(id); }
    [[nodiscard]] auto getUncondensedId(n_id_t id) const -> n_id_t { return m_inverse_map.at(id); }
    [[nodiscard]] auto getCondensedIds() const -> std::span< const n_id_t > { return m_condensed_ids; }

    [[nodiscard]] auto getLocalCondensedId(n_id_t node) const -> n_id_t
    {
        return static_cast< n_id_t >(std::distance(getCondensedIds().begin(),
                                                   std::ranges::lower_bound(getCondensedIds(), getCondensedId(node))));
    }
    template < el_o_t... orders >
    [[nodiscard]] auto getCondensedOwnedNodesView(const mesh::MeshPartition< orders... >& mesh) const
    {
        return getCondensedIds() |
               std::views::filter([&](n_id_t node) { return mesh.isOwnedNode(getUncondensedId(node)); });
    }
    template < el_o_t... orders >
    [[nodiscard]] auto getCondensedGhostNodesView(const mesh::MeshPartition< orders... >& mesh) const
    {
        return getCondensedIds() |
               std::views::filter([&](n_id_t node) { return mesh.isGhostNode(getUncondensedId(node)); });
    }

private:
    std::vector< n_id_t >                            m_condensed_ids;
    robin_hood::unordered_flat_map< n_id_t, n_id_t > m_forward_map, m_inverse_map;
};

template < CondensationPolicy CP, ProblemDef problem_def, el_o_t... orders >
auto makeCondensationMap(const MpiComm&                          comm,
                         const mesh::MeshPartition< orders... >& mesh,
                         util::ConstexprValue< problem_def >     probdef_ctwrpr,
                         CondensationPolicyTag< CP >             cp_tag = {}) -> NodeCondensationMap< CP >
{
    auto active_nodes = dofs::getActiveNodes(mesh, probdef_ctwrpr, cp_tag);
    activateOwned(comm, mesh, active_nodes);
    auto condensed_active_nodes = computeCondensedActiveNodeIds(comm, mesh, active_nodes);
    return {std::move(active_nodes), std::move(condensed_active_nodes)};
}
} // namespace dofs
} // namespace lstr
#endif // L3STER_DOFS_NODECONDENSATION_HPP
