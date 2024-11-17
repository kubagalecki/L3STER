#ifndef L3STER_DOFS_NODECONDENSATION_HPP
#define L3STER_DOFS_NODECONDENSATION_HPP

#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/common/Enums.hpp"
#include "l3ster/common/ProblemDefinition.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

namespace lstr
{
template < CondensationPolicy cond_policy >
struct CondensationPolicyTag
{
    static constexpr auto value = cond_policy;
};

inline constexpr auto no_condensation_tag  = CondensationPolicyTag< CondensationPolicy::None >{};
inline constexpr auto element_boundary_tag = CondensationPolicyTag< CondensationPolicy::ElementBoundary >{};

namespace dofs
{
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

template < CondensationPolicy >
class NodeCondensationMap
{
    using map_t = robin_hood::unordered_flat_map< n_id_t, n_id_t >;

public:
    template < RangeOfConvertibleTo_c< n_id_t > Uncond, RangeOfConvertibleTo_c< n_id_t > Cond >
    NodeCondensationMap(Uncond&& uncond, Cond&& cond)
    {
        util::throwingAssert(std::ranges::distance(uncond) == std::ranges::distance(cond));
        auto zipped   = std::views::zip(std::forward< Uncond >(uncond), std::forward< Cond >(cond));
        m_forward_map = map_t(zipped.begin(), zipped.end());
        for (auto&& [u, c] : zipped)
            m_inverse_map.emplace(c, u);
    }

    [[nodiscard]] auto getCondensedId(n_id_t id) const -> n_id_t { return m_forward_map.at(id); }
    [[nodiscard]] auto getUncondensedId(n_id_t id) const -> n_id_t { return m_inverse_map.at(id); }
    [[nodiscard]] auto getCondensedIds() const
    {
        return m_inverse_map | std::views::transform([](const auto& p) { return p.first; });
    }

    template < el_o_t... orders >
    [[nodiscard]] auto getCondensedOwnedNodesView(const mesh::MeshPartition< orders... >& mesh) const
    {
        const auto owned_pred = [&](n_id_t node) {
            return mesh.isOwnedNode(getUncondensedId(node));
        };
        return getCondensedIds() | std::views::filter(owned_pred);
    }
    template < el_o_t... orders >
    [[nodiscard]] auto getCondensedGhostNodesView(const mesh::MeshPartition< orders... >& mesh) const
    {
        const auto ghost_pred = [&](n_id_t node) {
            return mesh.isGhostNode(getUncondensedId(node));
        };
        return getCondensedIds() | std::views::filter(ghost_pred);
    }

private:
    map_t m_forward_map, m_inverse_map;
};

namespace detail
{
template < CondensationPolicy CP, ProblemDef problem_def, el_o_t... orders >
auto markActiveNodes(const mesh::MeshPartition< orders... >& mesh,
                     CondensationPolicyTag< CP >             cp_tag,
                     util::ConstexprValue< problem_def >) -> util::ArrayOwner< char >
{
    const auto g2l       = util::IndexMap{mesh.getAllNodes()};
    auto       retval    = util::ArrayOwner< char >(g2l.size(), false);
    const auto mark_node = [&](n_id_t node) {
        const auto lid = g2l(node);
        std::atomic_ref{retval[lid]}.store(true, std::memory_order_relaxed);
    };
    const auto mark_el_nodes = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& el) {
        for (auto n : getPrimaryNodesView(el, cp_tag))
            mark_node(n);
    };
    for (const auto& [domain, _] : problem_def)
        mesh.visit(mark_el_nodes, domain, std::execution::par);
    return retval;
}

inline void communicateActiveNodes(const MpiComm&                                       comm,
                                   std::shared_ptr< const comm::ImportExportContext<> > context,
                                   size_t                                               num_owned_nodes,
                                   util::ArrayOwner< char >&                            is_active)
{
    auto importer = comm::Import< char, local_dof_t >{context, 1};
    auto exporter = comm::Export< char, local_dof_t >{context, 1};
    exporter.setOwned(is_active | std::views::take(num_owned_nodes), is_active.size());
    exporter.setShared(is_active | std::views::drop(num_owned_nodes), is_active.size());
    importer.setOwned(is_active | std::views::take(num_owned_nodes), is_active.size());
    importer.setShared(is_active | std::views::drop(num_owned_nodes), is_active.size());
    exporter.doBlockingExport(comm, util::AtomicOrInto{});
    importer.doBlockingImport(comm);
}

struct OwnedCondensedNodes
{
    std::vector< n_id_t > uncondensed, condensed;
};
template < el_o_t... orders >
auto computeOwnedCondensedNodes(const MpiComm&                          comm,
                                const mesh::MeshPartition< orders... >& mesh,
                                const util::ArrayOwner< char >&         is_active) -> OwnedCondensedNodes
{
    auto retval            = OwnedCondensedNodes{};
    auto& [uncond, cond]   = retval;
    const auto owned_nodes = mesh.getOwnedNodes();
    for (auto&& [i, active] : is_active | std::views::enumerate | std::views::take(owned_nodes.size()))
        if (active)
            uncond.push_back(owned_nodes[i]);
    auto scan_val = std::views::single(uncond.size());
    comm.exclusiveScanInPlace(scan_val, MPI_SUM);
    const auto base_ind = comm.getRank() == 0 ? n_id_t{0} : static_cast< n_id_t >(scan_val.front());
    cond.resize(uncond.size());
    std::ranges::iota(cond, base_ind);
    return retval;
}

template < el_o_t... orders >
void appendGhosts(const MpiComm&                                       comm,
                  std::shared_ptr< const comm::ImportExportContext<> > context,
                  const mesh::MeshPartition< orders... >&              mesh,
                  const util::ArrayOwner< char >&                      is_active,
                  std::vector< n_id_t >&                               uncondensed,
                  std::vector< n_id_t >&                               condensed)
{
    using namespace std::views;
    const auto owned_nodes      = mesh.getOwnedNodes();
    const auto num_owned        = owned_nodes.size();
    auto       condensed_lookup = util::ArrayOwner< n_id_t >(is_active.size(), 0);
    for (auto&& [u, c] : zip(uncondensed, condensed))
        condensed_lookup.at(u - owned_nodes.front()) = c; // if this is executing then !owned_nodes.empty()
    auto importer = comm::Import< n_id_t, local_dof_t >{std::move(context), 1};
    importer.setOwned(condensed_lookup | take(num_owned), condensed_lookup.size());
    importer.setShared(condensed_lookup | drop(num_owned), condensed_lookup.size());
    importer.doBlockingImport(comm);
    for (auto&& [active, uncond, cond] : zip(is_active, mesh.getAllNodes(), condensed_lookup) | drop(num_owned))
        if (active)
        {
            uncondensed.push_back(uncond);
            condensed.push_back(cond);
        }
}
} // namespace detail

template < CondensationPolicy CP, ProblemDef problem_def, el_o_t... orders >
auto makeCondensationMap(const MpiComm&                          comm,
                         const mesh::MeshPartition< orders... >& mesh,
                         util::ConstexprValue< problem_def >     probdef_ctwrpr,
                         CondensationPolicyTag< CP >             cp_tag = {}) -> NodeCondensationMap< CP >
{
    const auto owned_nodes  = util::ArrayOwner< global_dof_t >{mesh.getOwnedNodes()};
    const auto shared_nodes = util::ArrayOwner< global_dof_t >{mesh.getGhostNodes()};
    auto       is_active    = detail::markActiveNodes(mesh, cp_tag, probdef_ctwrpr);
    using context_t         = const comm::ImportExportContext<>;
    const auto node_context = std::make_shared< context_t >(comm, std::span{owned_nodes}, std::span{shared_nodes});
    detail::communicateActiveNodes(comm, node_context, owned_nodes.size(), is_active);
    auto [active_uncond, active_cond] = detail::computeOwnedCondensedNodes(comm, mesh, is_active);
    detail::appendGhosts(comm, node_context, mesh, is_active, active_uncond, active_cond);
    return {std::move(active_uncond), std::move(active_cond)};
}
} // namespace dofs
} // namespace lstr
#endif // L3STER_DOFS_NODECONDENSATION_HPP
