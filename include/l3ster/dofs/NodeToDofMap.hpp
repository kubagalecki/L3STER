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
    template < CondensationPolicy CP, el_o_t... orders, ProblemDef problem_def >
    NodeToGlobalDofMap(const MpiComm&                          comm,
                       const mesh::MeshPartition< orders... >& m_mesh,
                       util::ConstexprValue< problem_def >     problemdef_ctwrpr,
                       const bcs::PeriodicBC< dofs_per_node >& periodic_bc = {},
                       CondensationPolicyTag< CP >             cp          = {})
        requires(problem_def.n_fields == dofs_per_node);

    [[nodiscard]] auto operator()(n_id_t node) const -> const payload_t& { return m_map.at(node); }
    [[nodiscard]] auto map() const -> const auto& { return m_map; }
    [[nodiscard]] auto ownership() const -> const auto& { return m_dof_ownership; }

private:
    inline void initFromDofs(const util::SegmentedOwnership< n_id_t >&                       node_ownership,
                             const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs,
                             const bcs::PeriodicBC< dofs_per_node >&                         periodic_bc,
                             global_dof_t                                                    base_dof,
                             size_t                                                          num_owned_dofs,
                             const std::set< global_dof_t >&                                 shared_dofs);

    robin_hood::unordered_flat_map< n_id_t, payload_t > m_map;
    util::SegmentedOwnership< global_dof_t >            m_dof_ownership;
};
template < CondensationPolicy CP, el_o_t... orders, ProblemDef problem_def >
NodeToGlobalDofMap(const MpiComm&,
                   const mesh::MeshPartition< orders... >&,
                   util::ConstexprValue< problem_def >,
                   const bcs::PeriodicBC< problem_def.n_fields >&,
                   CondensationPolicyTag< CP >) -> NodeToGlobalDofMap< problem_def.n_fields >;
template < el_o_t... orders, ProblemDef problem_def >
NodeToGlobalDofMap(const MpiComm&, const mesh::MeshPartition< orders... >&, util::ConstexprValue< problem_def >)
    -> NodeToGlobalDofMap< problem_def.n_fields >;

template < size_t dofs_per_node, size_t num_maps >
class NodeToLocalDofMap
{
    using payload_t = std::array< std::array< local_dof_t, dofs_per_node >, num_maps >;
    using map_t     = robin_hood::unordered_flat_map< n_id_t, payload_t >;

public:
    using dof_t = local_dof_t;
    static constexpr bool isValid(dof_t dof) { return dof != invalid_local_dof; }

    NodeToLocalDofMap() = default;
    inline NodeToLocalDofMap(const NodeToGlobalDofMap< dofs_per_node >& global_map,
                             const std::same_as< tpetra_map_t > auto&... local_global_maps)
        requires(sizeof...(local_global_maps) == num_maps);
    [[nodiscard]] const payload_t& operator()(n_id_t node) const noexcept { return m_map.at(node); }

    [[nodiscard]] auto size() const -> size_t { return m_map.size(); }
    [[nodiscard]] auto begin() const { return m_map.cbegin(); }
    [[nodiscard]] auto end() const { return m_map.cend(); }

private:
    map_t m_map;
};

template < size_t dofs_per_node >
NodeToLocalDofMap(const NodeToGlobalDofMap< dofs_per_node >& global_map,
                  const std::same_as< tpetra_map_t > auto&... local_global_maps)
    -> NodeToLocalDofMap< dofs_per_node, sizeof...(local_global_maps) >;

template < size_t max_dofs_per_node >
class LocalDofMap
{
public:
    using dof_t     = local_dof_t;
    using payload_t = std::array< dof_t, max_dofs_per_node >;
    static constexpr bool isValid(dof_t dof) { return dof != invalid_local_dof; }

    LocalDofMap() = default;
    template < el_o_t... orders >
    LocalDofMap(const NodeToGlobalDofMap< max_dofs_per_node >& global_map,
                const mesh::MeshPartition< orders... >&        mesh);

    [[nodiscard]] auto   operator()(n_loc_id_t node) const -> const payload_t& { return m_map.at(node); }
    [[nodiscard]] size_t getNumOwnedDofs() const { return static_cast< size_t >(m_num_owned); }
    [[nodiscard]] size_t getNumSharedDofs() const { return static_cast< size_t >(m_num_total - m_num_owned); }
    [[nodiscard]] size_t getNumTotalDofs() const { return static_cast< size_t >(m_num_total); }

    [[nodiscard]] auto size() const -> size_t { return m_map.size(); }
    [[nodiscard]] auto begin() const { return m_map.cbegin(); }
    [[nodiscard]] auto end() const { return m_map.cend(); }

private:
    util::ArrayOwner< payload_t > m_map;
    dof_t                         m_num_owned{}, m_num_total{};
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
NodeToLocalDofMap< dofs_per_node, num_maps >::NodeToLocalDofMap(
    const NodeToGlobalDofMap< dofs_per_node >& global_map,
    const std::same_as< tpetra_map_t > auto&... local_global_maps)
    requires(sizeof...(local_global_maps) == num_maps)
    : m_map{global_map.map().size()}
{
    L3STER_PROFILE_FUNCTION;
    const auto get_node_dofs = [&](const NodeToGlobalDofMap< dofs_per_node >::payload_t& gids,
                                   const tpetra_map_t&                                   map) {
        auto retval = std::array< local_dof_t, dofs_per_node >{};
        std::ranges::transform(gids, retval.begin(), [&](global_dof_t dof) {
            return dof == invalid_global_dof ? invalid_local_dof : map.getLocalElement(dof);
        });
        return retval;
    };
    for (const auto& [node, gids] : global_map.map())
        m_map[node] = payload_t{get_node_dofs(gids, local_global_maps)...};
}

template < size_t max_dofs_per_node >
template < el_o_t... orders >
LocalDofMap< max_dofs_per_node >::LocalDofMap(const NodeToGlobalDofMap< max_dofs_per_node >& global_map,
                                              const mesh::MeshPartition< orders... >&        mesh)
    : m_map(global_map.map().size()),
      m_num_owned{static_cast< dof_t >(global_map.ownership().owned().size())},
      m_num_total{static_cast< dof_t >(global_map.ownership().localSize())}
{
    L3STER_PROFILE_FUNCTION;
    const auto translate_dof = [&](global_dof_t gid) {
        return gid != invalid_global_dof ? global_map.ownership().getLocalIndex(gid) : invalid_local_dof;
    };
    for (const auto& [node_gid, gids] : global_map.map())
    {
        const auto node_lid = mesh.getLocalNodeIndex(node_gid);
        auto       lids     = payload_t{};
        std::ranges::transform(gids, lids.begin(), translate_dof);
        m_map.at(node_lid) = lids;
    }
}

namespace detail
{
template < el_o_t... orders, size_t max_dofs_per_node >
auto makeNodeOwnership(const mesh::MeshPartition< orders... >&     mesh,
                       const bcs::PeriodicBC< max_dofs_per_node >& periodic_bc) -> util::SegmentedOwnership< n_id_t >
{
    const auto& mesh_ownership = mesh.getNodeOwnership();
    const auto  num_shared     = mesh_ownership.shared().size() + periodic_bc.getPeriodicGhosts().size();
    auto        shared         = util::ArrayOwner< n_id_t >(num_shared);
    const auto  write_iter     = std::ranges::copy(mesh_ownership.shared(), shared.begin()).out;
    std::ranges::copy(periodic_bc.getPeriodicGhosts(), write_iter);
    const auto owned_begin = mesh_ownership.owned().empty() ? 0uz : mesh_ownership.owned().front();
    return {owned_begin, mesh_ownership.owned().size(), shared};
}

template < CondensationPolicy CP, el_o_t... orders, size_t num_domains, size_t max_dofs_per_node >
auto makeLocalDofBmp(const mesh::MeshPartition< orders... >&             mesh,
                     const ProblemDef< num_domains, max_dofs_per_node >& problem_def,
                     const util::SegmentedOwnership< n_id_t >&           node_ownership,
                     const bcs::PeriodicBC< max_dofs_per_node >&         periodic_bc,
                     CondensationPolicyTag< CP > = {}) -> Kokkos::View< char**, Kokkos::LayoutLeft >
{
    auto retval = Kokkos::View< char**, Kokkos::LayoutLeft >{"DOF bmp", node_ownership.localSize(), max_dofs_per_node};
    for (const auto& [domain, active_bmp] : problem_def)
    {
        const auto active_inds = util::getTrueInds(active_bmp);
        const auto mark_node   = [&](n_id_t node) {
            const auto lid           = node_ownership.getLocalIndex(node);
            const auto periodic_info = periodic_bc.lookup(node);
            for (auto j : active_inds)
            {
                const auto periodic   = periodic_info[j];
                const auto actual_lid = periodic == invalid_node ? lid : node_ownership.getLocalIndex(periodic);
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

template < size_t max_dofs_per_node >
auto computeNumOwnedDofs(const util::SegmentedOwnership< n_id_t >&               node_ownership,
                         const bcs::PeriodicBC< max_dofs_per_node >&             periodic_bc,
                         const Kokkos::View< const char**, Kokkos::LayoutLeft >& dof_bmp) -> size_t
{
    size_t retval = 0;
    for (size_t i = 0; auto node : node_ownership.owned()) // views::enumerate yields decltype(i)==__int128
    {
        const auto periodic_info = periodic_bc.lookup(node);
        for (auto&& [j, periodic_node] : periodic_info | std::views::enumerate)
            retval += periodic_node == invalid_node and dof_bmp(i, j);
        ++i;
    }
    return retval;
}

inline auto computeBaseDof(const MpiComm& comm, size_t num_owned_dofs) -> global_dof_t
{
    auto scan_view = std::views::single(num_owned_dofs);
    comm.exclusiveScanInPlace(scan_view, MPI_SUM);
    return static_cast< global_dof_t >(comm.getRank() == 0 ? 0 : scan_view.front());
}

template < size_t max_dofs_per_node >
auto computeOwnedDofs(const util::SegmentedOwnership< n_id_t >&               node_ownership,
                      const bcs::PeriodicBC< max_dofs_per_node >&             periodic_bc,
                      const Kokkos::View< const char**, Kokkos::LayoutLeft >& dof_bmp,
                      n_id_t base_dof) -> Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >
{
    auto retval = Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >{"DOFs", dof_bmp.layout()};
    for (size_t i = 0; auto node : node_ownership.owned()) // views::enumerate yields decltype(i)==__int128
    {
        const auto periodic_info = periodic_bc.lookup(node);
        for (auto&& [j, periodic_node] : periodic_info | std::views::enumerate)
            retval(i, j) = periodic_node == invalid_node and dof_bmp(i, j) ? base_dof++ : invalid_global_dof;
        ++i;
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

template < size_t max_dofs_per_node >
auto fixPeriodicDofs(const util::SegmentedOwnership< n_id_t >&                 node_ownership,
                     const Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >& dofs,
                     const bcs::PeriodicBC< max_dofs_per_node >&               periodic_bc)
{
    for (const auto& [gid, periodic] : periodic_bc)
    {
        const auto lid = node_ownership.getLocalIndex(gid);
        for (auto&& [i, periodic_node] : periodic | std::views::enumerate)
            if (periodic_node != invalid_node)
            {
                const auto periodic_lid = node_ownership.getLocalIndex(periodic_node);
                dofs(lid, i)            = dofs(periodic_lid, i);
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
void NodeToGlobalDofMap< dofs_per_node >::initFromDofs(
    const util::SegmentedOwnership< n_id_t >&                       node_ownership,
    const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs,
    const bcs::PeriodicBC< dofs_per_node >&                         periodic_bc,
    global_dof_t                                                    base_dof,
    size_t                                                          num_owned_dofs,
    const std::set< global_dof_t >&                                 shared_dofs)
{
    const auto insert_node = [&](n_id_t node, auto lid) {
        auto node_dofs = payload_t{};
        for (auto&& [j, node_dof] : node_dofs | std::views::enumerate)
            node_dof = dofs(lid, j);
        m_map[node] = node_dofs;
    };
    for (size_t i = 0; auto node : node_ownership.owned()) // views::enumerate yields decltype(i)==__int128
        insert_node(node, i++);
    const auto num_owned_nodes = static_cast< ptrdiff_t >(node_ownership.owned().size());
    for (auto&& [i, node] : node_ownership.shared() | std::views::enumerate)
        if (not std::ranges::binary_search(periodic_bc.getPeriodicGhosts(), node))
            insert_node(node, num_owned_nodes + i);
    m_dof_ownership = {base_dof, num_owned_dofs, shared_dofs};
}

template < size_t dofs_per_node >
template < CondensationPolicy CP, el_o_t... orders, ProblemDef problem_def >
NodeToGlobalDofMap< dofs_per_node >::NodeToGlobalDofMap(const MpiComm&                          comm,
                                                        const mesh::MeshPartition< orders... >& mesh,
                                                        util::ConstexprValue< problem_def >,
                                                        const bcs::PeriodicBC< dofs_per_node >& periodic_bc,
                                                        CondensationPolicyTag< CP >             cp)
    requires(problem_def.n_fields == dofs_per_node)
{
    L3STER_PROFILE_FUNCTION;
    const auto node_ownership  = detail::makeNodeOwnership(mesh, periodic_bc);
    const auto num_owned_nodes = node_ownership.owned().size();
    const auto dof_bmp         = detail::makeLocalDofBmp(mesh, problem_def, node_ownership, periodic_bc, cp);
    const auto context         = node_ownership.makeCommContext(comm);
    detail::exportDofBmp(comm, context, dof_bmp, num_owned_nodes);
    const auto num_owned_dofs = detail::computeNumOwnedDofs(node_ownership, periodic_bc, dof_bmp);
    const auto base_dof       = detail::computeBaseDof(comm, num_owned_dofs);
    auto       dofs           = detail::computeOwnedDofs(node_ownership, periodic_bc, dof_bmp, base_dof);
    detail::communicateSharedDofs(comm, context, dofs, num_owned_nodes);
    detail::fixPeriodicDofs(node_ownership, dofs, periodic_bc);
    detail::communicateSharedDofs(comm, context, dofs, num_owned_nodes);
    const auto shared_dofs = detail::computeSharedDofs(dofs, base_dof, num_owned_dofs);
    initFromDofs(node_ownership, dofs, periodic_bc, base_dof, num_owned_dofs, shared_dofs);
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_NODETODOFMAP_HPP
