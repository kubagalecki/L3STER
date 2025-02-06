#ifndef L3STER_DOFS_NODETODOFMAP_HPP
#define L3STER_DOFS_NODETODOFMAP_HPP

#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/dofs/NodeCondensation.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/SegmentedOwnership.hpp"

namespace lstr::dofs
{
template < size_t dofs_per_node >
class NodeToGlobalDofMap
{
public:
    using dof_t     = global_dof_t;
    using payload_t = std::array< global_dof_t, dofs_per_node >;
    static constexpr bool isValid(dof_t dof) { return dof != invalid_global_dof; }

    NodeToGlobalDofMap() = default;
    template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
    NodeToGlobalDofMap(const MpiComm&                          comm,
                       const mesh::MeshPartition< orders... >& m_mesh,
                       const NodeCondensationMap< CP >&        cond_map,
                       util::ConstexprValue< problem_def >     problemdef_ctwrpr,
                       const bcs::PeriodicBC< dofs_per_node >& periodic_bc = {})
        requires(problem_def.n_fields == dofs_per_node);

    [[nodiscard]] auto operator()(n_id_t node) const -> const payload_t& { return m_map.at(node); }
    [[nodiscard]] auto ownership() const -> const auto& { return m_dof_ownership; }

private:
    template < CondensationPolicy CP >
    void initFromDofs(std::span< const n_id_t >                                       nodes,
                      size_t                                                          num_owned,
                      const NodeCondensationMap< CP >&                                cond_map,
                      const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs,
                      const bcs::PeriodicBC< dofs_per_node >&                         periodic_bc,
                      global_dof_t                                                    base_dof,
                      size_t                                                          num_owned_dofs,
                      const std::set< global_dof_t >&                                 shared_dofs);

    robin_hood::unordered_flat_map< n_id_t, payload_t > m_map;
    util::SegmentedOwnership< global_dof_t >            m_dof_ownership;
};
template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
NodeToGlobalDofMap(const MpiComm&,
                   const mesh::MeshPartition< orders... >&,
                   const NodeCondensationMap< CP >&,
                   util::ConstexprValue< problem_def >,
                   const bcs::PeriodicBC< problem_def.n_fields >&) -> NodeToGlobalDofMap< problem_def.n_fields >;
template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
NodeToGlobalDofMap(const MpiComm&,
                   const mesh::MeshPartition< orders... >&,
                   const NodeCondensationMap< CP >&,
                   util::ConstexprValue< problem_def >) -> NodeToGlobalDofMap< problem_def.n_fields >;

template < size_t dofs_per_node, size_t num_maps >
class NodeToLocalDofMap
{
    using payload_t = std::array< std::array< local_dof_t, dofs_per_node >, num_maps >;
    using map_t     = robin_hood::unordered_flat_map< n_id_t, payload_t >;

public:
    using dof_t = local_dof_t;
    static constexpr bool isValid(dof_t dof) { return dof != invalid_local_dof; }

    NodeToLocalDofMap() = default;
    template < CondensationPolicy CP >
    NodeToLocalDofMap(const NodeCondensationMap< CP >&           cond_map,
                      const NodeToGlobalDofMap< dofs_per_node >& global_map,
                      const std::same_as< tpetra_map_t > auto&... local_global_maps)
        requires(sizeof...(local_global_maps) == num_maps);
    [[nodiscard]] const payload_t& operator()(n_id_t node) const noexcept { return m_map.at(node); }

    [[nodiscard]] auto size() const -> size_t { return m_map.size(); }
    [[nodiscard]] auto begin() const { return m_map.cbegin(); }
    [[nodiscard]] auto end() const { return m_map.cend(); }

private:
    map_t m_map;
};

template < CondensationPolicy CP, size_t dofs_per_node >
NodeToLocalDofMap(const NodeCondensationMap< CP >&           cond_map,
                  const NodeToGlobalDofMap< dofs_per_node >& global_map,
                  const std::same_as< tpetra_map_t > auto&... local_global_maps)
    -> NodeToLocalDofMap< dofs_per_node, sizeof...(local_global_maps) >;

template < size_t max_dofs_per_node >
class LocalDofMap
{
public:
    using dof_t = local_dof_t;
    static constexpr bool isValid(dof_t dof) { return dof != invalid_local_dof; }

    LocalDofMap() = default;
    template < el_o_t... orders >
    LocalDofMap(const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                const NodeToGlobalDofMap< max_dofs_per_node >&         global_map,
                const mesh::MeshPartition< orders... >&                mesh);

    [[nodiscard]] auto   operator()(n_loc_id_t node) const -> const auto& { return m_map.at(node); }
    [[nodiscard]] size_t getNumOwnedDofs() const { return static_cast< size_t >(m_num_owned); }
    [[nodiscard]] size_t getNumSharedDofs() const { return static_cast< size_t >(m_num_total - m_num_owned); }
    [[nodiscard]] size_t getNumTotalDofs() const { return static_cast< size_t >(m_num_total); }

    [[nodiscard]] auto size() const -> size_t { return m_map.size(); }
    [[nodiscard]] auto begin() const { return m_map.cbegin(); }
    [[nodiscard]] auto end() const { return m_map.cend(); }

private:
    util::ArrayOwner< std::array< dof_t, max_dofs_per_node > > m_map;
    dof_t                                                      m_num_owned{}, m_num_total{};
};

namespace detail
{
template < typename T >
inline constexpr bool is_node_map = false;
template < size_t dpn >
inline constexpr bool is_node_map< NodeToGlobalDofMap< dpn > > = true;
template < size_t dpn, size_t nm >
inline constexpr bool is_node_map< NodeToLocalDofMap< dpn, nm > > = true;
} // namespace detail

template < typename T >
concept NodeToDofMap_c = detail::is_node_map< T >;

template < size_t dofs_per_node, size_t num_maps >
template < CondensationPolicy CP >
NodeToLocalDofMap< dofs_per_node, num_maps >::NodeToLocalDofMap(
    const NodeCondensationMap< CP >&           cond_map,
    const NodeToGlobalDofMap< dofs_per_node >& global_map,
    const std::same_as< tpetra_map_t > auto&... local_global_maps)
    requires(sizeof...(local_global_maps) == num_maps)
    : m_map(std::ranges::size(cond_map.getCondensedIds()))
{
    L3STER_PROFILE_FUNCTION;
    const auto get_node_dofs = [&](n_id_t node, const tpetra_map_t& map) {
        auto retval = std::array< local_dof_t, dofs_per_node >{};
        std::ranges::transform(global_map(node), retval.begin(), [&](global_dof_t dof) {
            return NodeToGlobalDofMap< dofs_per_node >::isValid(dof) ? map.getLocalElement(dof) : invalid_local_dof;
        });
        return retval;
    };
    for (n_id_t cond_node : cond_map.getCondensedIds())
        m_map[cond_map.getUncondensedId(cond_node)] = payload_t{get_node_dofs(cond_node, local_global_maps)...};
}

template < size_t max_dofs_per_node >
template < el_o_t... orders >
LocalDofMap< max_dofs_per_node >::LocalDofMap(const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                                              const NodeToGlobalDofMap< max_dofs_per_node >&         global_map,
                                              const mesh::MeshPartition< orders... >&                mesh)
    : m_map(std::ranges::size(cond_map.getCondensedIds())),
      m_num_owned{static_cast< dof_t >(global_map.ownership().owned().size())},
      m_num_total{static_cast< dof_t >(m_num_owned + global_map.ownership().shared().size())}
{
    L3STER_PROFILE_FUNCTION;
    const auto translate_dof = [&](global_dof_t gid) {
        return gid != invalid_global_dof ? global_map.ownership().getLocalIndex(gid) : invalid_local_dof;
    };
    for (auto cond_node : cond_map.getCondensedIds())
    {
        const auto& src_gids    = global_map(cond_node);
        const auto  uncond_node = cond_map.getUncondensedId(cond_node);
        const auto  local_node  = mesh.getLocalNodeIndex(uncond_node);
        auto&       dest_lids   = m_map.at(local_node);
        std::ranges::transform(src_gids, dest_lids.begin(), translate_dof);
    }
}

namespace detail
{
struct CondensedNodes
{
    std::vector< n_id_t > nodes;
    size_t                num_owned_nodes;
};
template < el_o_t... orders, CondensationPolicy CP, size_t max_dofs_per_node = 0 >
auto getCondensedNodes(const mesh::MeshPartition< orders... >&     mesh,
                       const NodeCondensationMap< CP >&            cond_map,
                       const bcs::PeriodicBC< max_dofs_per_node >& periodic) -> CondensedNodes
{
    auto retval              = CondensedNodes{};
    auto& [nodes, num_owned] = retval;
    const auto copy_cond     = [&](auto&& uncond_range) {
        for (auto node_uncond : std::forward< decltype(uncond_range) >(uncond_range))
            if (const auto maybe_cond = cond_map.getCondensedIdOpt(node_uncond); maybe_cond.has_value())
                nodes.push_back(*maybe_cond);
    };
    copy_cond(mesh.getOwnedNodes());
    std::ranges::sort(nodes);
    num_owned = nodes.size();
    copy_cond(mesh.getGhostNodes());
    copy_cond(periodic.getPeriodicGhosts());
    std::ranges::sort(nodes | std::views::drop(num_owned));
    util::throwingAssert(std::ranges::none_of(
        nodes | std::views::drop(num_owned) | std::views::adjacent_transform< 2 >(std::equal_to{}), std::identity{}));
    return retval;
}

template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
auto makeLocalDofBmp(const mesh::MeshPartition< orders... >& mesh,
                     const NodeCondensationMap< CP >&        cond_map,
                     const util::IndexMap< n_id_t >&         g2l,
                     util::ConstexprValue< problem_def >,
                     const bcs::PeriodicBC< problem_def.n_fields >& periodic_bc)
    -> Kokkos::View< char**, Kokkos::LayoutLeft >
{
    const auto num_all = g2l.size();
    auto       retval  = Kokkos::View< char**, Kokkos::LayoutLeft >{"DOF bmp", num_all, problem_def.n_fields};
    for (const auto& [domain, active_bmp] : problem_def)
    {
        const auto active_inds = util::getTrueInds(active_bmp);
        const auto mark_node   = [&](n_id_t node) {
            const auto cond_lid      = g2l(cond_map.getCondensedId(node));
            const auto periodic_info = periodic_bc.lookup(node);
            for (auto j : active_inds)
            {
                const auto periodic   = periodic_info[j];
                const auto actual_lid = periodic != invalid_node ? g2l(cond_map.getCondensedId(periodic)) : cond_lid;
                std::atomic_ref{retval(actual_lid, j)}.store(true, std::memory_order_relaxed);
            }
        };
        const auto mark_el = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& el) {
            for (auto n : getPrimaryNodesView< CP >(el))
                mark_node(n);
        };
        mesh.visit(mark_el, domain, std::execution::par);
    }
    return retval;
}

inline auto makeCondensedCommContext(const MpiComm& comm, std::span< const n_id_t > nodes, size_t num_owned_nodes)
    -> std::shared_ptr< const comm::ImportExportContext<> >
{
    const auto nodes_signed = util::ArrayOwner< global_dof_t >{nodes};
    const auto owned        = std::span{nodes_signed | std::views::take(num_owned_nodes)};
    const auto shared       = std::span{nodes_signed | std::views::drop(num_owned_nodes)};
    return std::make_shared< const comm::ImportExportContext<> >(comm, owned, shared);
}

inline void exportDofBmp(const MpiComm&                                              comm,
                         const std::shared_ptr< const comm::ImportExportContext<> >& context,
                         const Kokkos::View< char**, Kokkos::LayoutLeft >&           dof_bmp,
                         size_t                                                      num_owned_nodes)
{
    using namespace std::views;
    const auto num_all       = dof_bmp.extent(0);
    const auto dofs_per_node = dof_bmp.extent(1);
    auto       exporter      = comm::Export< char, local_dof_t >{context, dofs_per_node};
    const auto shared_range  = std::make_pair(num_owned_nodes, num_all);
    const auto shared_view   = Kokkos::subview(dof_bmp, shared_range, Kokkos::ALL());
    exporter.setOwned(dof_bmp);
    exporter.setShared(shared_view);
    exporter.doBlockingExport(comm, util::AtomicOrInto{});
}

template < CondensationPolicy CP, size_t max_dofs_per_node >
auto computeNumOwnedDofs(std::span< const n_id_t >                               cond_nodes,
                         size_t                                                  num_owned_nodes,
                         const NodeCondensationMap< CP >&                        cond_map,
                         const bcs::PeriodicBC< max_dofs_per_node >&             periodic_bc,
                         const Kokkos::View< const char**, Kokkos::LayoutLeft >& dof_bmp) -> size_t
{
    size_t retval = 0;
    for (auto&& [i, cond_node] : cond_nodes | std::views::take(num_owned_nodes) | std::views::enumerate)
    {
        const auto uncond_node   = cond_map.getUncondensedId(cond_node);
        const auto periodic_info = periodic_bc.lookup(uncond_node);
        for (auto&& [j, periodic_node] : periodic_info | std::views::enumerate)
            retval += periodic_node == invalid_node and dof_bmp(i, j);
    }
    return retval;
}

inline auto computeBaseDof(const MpiComm& comm, size_t num_owned_dofs) -> global_dof_t
{
    auto scan_view = std::views::single(num_owned_dofs);
    comm.exclusiveScanInPlace(scan_view, MPI_SUM);
    return static_cast< global_dof_t >(comm.getRank() == 0 ? 0 : scan_view.front());
}

template < CondensationPolicy CP, size_t max_dofs_per_node >
auto computeOwnedDofs(std::span< const n_id_t >                               cond_nodes,
                      size_t                                                  num_owned_nodes,
                      const NodeCondensationMap< CP >&                        cond_map,
                      const bcs::PeriodicBC< max_dofs_per_node >&             periodic_bc,
                      const Kokkos::View< const char**, Kokkos::LayoutLeft >& dof_bmp,
                      n_id_t base_dof) -> Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >
{
    auto retval = Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >{"DOFs", dof_bmp.layout()};
    for (auto&& [i, cond_node] : cond_nodes | std::views::take(num_owned_nodes) | std::views::enumerate)
    {
        const auto uncond_node   = cond_map.getUncondensedId(cond_node);
        const auto periodic_info = periodic_bc.lookup(uncond_node);
        for (auto&& [j, periodic_node] : periodic_info | std::views::enumerate)
            retval(i, j) = periodic_node == invalid_node and dof_bmp(i, j) ? base_dof++ : invalid_global_dof;
    }
    return retval;
}

inline void communicateSharedDofs(const MpiComm&                                              comm,
                                  const std::shared_ptr< const comm::ImportExportContext<> >& context,
                                  const Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >&   dofs,
                                  size_t                                                      num_owned_nodes)
{
    const auto num_all       = dofs.extent(0);
    const auto dofs_per_node = dofs.extent(1);
    auto       importer      = comm::Import< global_dof_t, local_dof_t >{context, dofs_per_node};
    const auto shared_range  = std::make_pair(num_owned_nodes, num_all);
    const auto shared_view   = Kokkos::subview(dofs, shared_range, Kokkos::ALL());
    importer.setOwned(dofs);
    importer.setShared(shared_view);
    importer.doBlockingImport(comm);
}

template < CondensationPolicy CP, size_t max_dofs_per_node >
auto fixPeriodicDofs(const NodeCondensationMap< CP >&                          cond_map,
                     const util::IndexMap< n_id_t >&                           g2l,
                     const Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >& dofs,
                     const bcs::PeriodicBC< max_dofs_per_node >&               periodic_bc)
{
    for (const auto& [gid, periodic] : periodic_bc)
    {
        if (const auto maybe_cond = cond_map.getCondensedIdOpt(gid); maybe_cond.has_value())
        {
            const auto lid = g2l(*maybe_cond);
            for (auto&& [i, periodic_node] : periodic | std::views::enumerate)
                if (periodic_node != invalid_node)
                {
                    const auto periodic_lid = g2l(cond_map.getCondensedId(periodic_node));
                    dofs(lid, i)            = dofs(periodic_lid, i);
                }
        }
    }
}

inline auto computeSharedDofs(const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs,
                              global_dof_t                                                    base_dof,
                              size_t num_owned_dofs) -> std::set< global_dof_t >
{
    const auto dof_bound = base_dof + static_cast< global_dof_t >(num_owned_dofs);
    auto       retval    = std::set< global_dof_t >{};
    for (auto dof : util::flatten(dofs))
        if (dof != invalid_global_dof and not(dof >= base_dof and dof < dof_bound))
            retval.insert(dof);
    return retval;
}
} // namespace detail

template < size_t dofs_per_node >
template < CondensationPolicy CP >
void NodeToGlobalDofMap< dofs_per_node >::initFromDofs(
    std::span< const n_id_t >                                       cond_nodes,
    size_t                                                          num_owned_nodes,
    const NodeCondensationMap< CP >&                                cond_map,
    const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs,
    const bcs::PeriodicBC< dofs_per_node >&                         periodic_bc,
    global_dof_t                                                    base_dof,
    size_t                                                          num_owned_dofs,
    const std::set< global_dof_t >&                                 shared_dofs)
{
    const auto insert_node = [&](n_id_t cond_node, auto lid) {
        auto node_dofs = payload_t{};
        for (auto&& [j, node_dof] : node_dofs | std::views::enumerate)
            node_dof = dofs(lid, j);
        m_map[cond_node] = node_dofs;
    };
    for (auto&& [i, cond_node] : cond_nodes | std::views::enumerate | std::views::take(num_owned_nodes))
        insert_node(cond_node, i);
    for (auto&& [i, cond_node] : cond_nodes | std::views::enumerate | std::views::drop(num_owned_nodes))
        if (not std::ranges::binary_search(periodic_bc.getPeriodicGhosts(), cond_map.getUncondensedId(cond_node)))
            insert_node(cond_node, i);
    m_dof_ownership = {base_dof, num_owned_dofs, shared_dofs};
}

template < size_t dofs_per_node >
template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
NodeToGlobalDofMap< dofs_per_node >::NodeToGlobalDofMap(const MpiComm&                          comm,
                                                        const mesh::MeshPartition< orders... >& mesh,
                                                        const NodeCondensationMap< CP >&        cond_map,
                                                        util::ConstexprValue< problem_def >     probdef_ctwrpr,
                                                        const bcs::PeriodicBC< dofs_per_node >& periodic_bc)
    requires(problem_def.n_fields == dofs_per_node)
{
    L3STER_PROFILE_FUNCTION;
    const auto [nodes, num_owned_nodes] = detail::getCondensedNodes(mesh, cond_map, periodic_bc);
    const auto g2l                      = util::IndexMap{nodes};
    const auto dof_bmp                  = detail::makeLocalDofBmp(mesh, cond_map, g2l, probdef_ctwrpr, periodic_bc);
    const auto context                  = detail::makeCondensedCommContext(comm, nodes, num_owned_nodes);
    detail::exportDofBmp(comm, context, dof_bmp, num_owned_nodes);
    const auto num_owned_dofs = detail::computeNumOwnedDofs(nodes, num_owned_nodes, cond_map, periodic_bc, dof_bmp);
    const auto base_dof       = detail::computeBaseDof(comm, num_owned_dofs);
    auto       dofs = detail::computeOwnedDofs(nodes, num_owned_nodes, cond_map, periodic_bc, dof_bmp, base_dof);
    detail::communicateSharedDofs(comm, context, dofs, num_owned_nodes);
    detail::fixPeriodicDofs(cond_map, g2l, dofs, periodic_bc);
    detail::communicateSharedDofs(comm, context, dofs, num_owned_nodes);
    const auto shared_dofs = detail::computeSharedDofs(dofs, base_dof, num_owned_dofs);
    initFromDofs(nodes, num_owned_nodes, cond_map, dofs, periodic_bc, base_dof, num_owned_dofs, shared_dofs);
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_NODETODOFMAP_HPP
