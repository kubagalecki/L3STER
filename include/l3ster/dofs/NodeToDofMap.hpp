#ifndef L3STER_DOFS_NODETODOFMAP_HPP
#define L3STER_DOFS_NODETODOFMAP_HPP

#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/dofs/NodeCondensation.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/IndexMap.hpp"

namespace lstr::dofs
{
template < size_t dofs_per_node >
class NodeToGlobalDofMap
{

public:
    using dof_t                               = global_dof_t;
    using payload_t                           = std::array< global_dof_t, dofs_per_node >;
    static constexpr global_dof_t invalid_dof = -1;
    static constexpr bool         isValid(dof_t dof) { return dof != invalid_dof; }

    NodeToGlobalDofMap() = default;
    template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
    NodeToGlobalDofMap(const MpiComm&                          comm,
                       const mesh::MeshPartition< orders... >& m_mesh,
                       const NodeCondensationMap< CP >&        cond_map,
                       util::ConstexprValue< problem_def >     problemdef_ctwrpr)
        requires(problem_def.n_fields == dofs_per_node);

    [[nodiscard]] auto operator()(n_id_t node) const -> const payload_t& { return m_map.at(node); }
    [[nodiscard]] auto getNumOwnedDofs() const { return m_num_owned; }

private:
    inline void initFromDofs(std::span< const n_id_t >                                       cond_nodes,
                             size_t                                                          num_owned,
                             const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs);

    using map_t = robin_hood::unordered_flat_map< n_id_t, payload_t >;
    map_t  m_map;
    size_t m_num_owned = 0;
};

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
    using dof_t                        = local_dof_t;
    static constexpr dof_t invalid_dof = -1;
    static constexpr bool  isValid(dof_t dof) { return dof != invalid_dof; }

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
    using dof_t                        = local_dof_t;
    static constexpr dof_t invalid_dof = -1;
    static constexpr bool  isValid(dof_t dof) { return dof != invalid_dof; }

    LocalDofMap() = default;
    LocalDofMap(const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                const NodeToGlobalDofMap< max_dofs_per_node >&         global_map,
                const mesh::NodeMap&                                   node_map,
                std::span< const global_dof_t >                        all_dofs,
                size_t                                                 num_owned_dofs);

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
            return NodeToGlobalDofMap< dofs_per_node >::isValid(dof) ? map.getLocalElement(dof) : invalid_dof;
        });
        return retval;
    };
    for (n_id_t cond_node : cond_map.getCondensedIds())
        m_map[cond_map.getUncondensedId(cond_node)] = payload_t{get_node_dofs(cond_node, local_global_maps)...};
}

template < size_t max_dofs_per_node >
LocalDofMap< max_dofs_per_node >::LocalDofMap(const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                                              const NodeToGlobalDofMap< max_dofs_per_node >&         global_map,
                                              const mesh::NodeMap&                                   node_map,
                                              std::span< const global_dof_t >                        all_dofs,
                                              size_t                                                 num_owned_dofs)
    : m_map(std::ranges::size(cond_map.getCondensedIds())),
      m_num_owned{static_cast< dof_t >(num_owned_dofs)},
      m_num_total{static_cast< dof_t >(all_dofs.size())}
{
    L3STER_PROFILE_FUNCTION;
    auto       g2l           = util::IndexMap< global_dof_t, local_dof_t >{all_dofs};
    const auto translate_dof = [&g2l](global_dof_t gid) {
        return NodeToGlobalDofMap< max_dofs_per_node >::isValid(gid) ? g2l(gid) : invalid_dof;
    };
    for (auto cond_node : cond_map.getCondensedIds())
    {
        const auto& src_gids    = global_map(cond_node);
        const auto  uncond_node = cond_map.getUncondensedId(cond_node);
        const auto  local_node  = node_map.toLocal(uncond_node);
        auto&       dest_lids   = m_map.at(local_node);
        std::ranges::transform(src_gids, dest_lids.begin(), translate_dof);
    }
}

namespace detail
{
struct CondensedNodes
{
    std::vector< n_id_t > nodes;
    size_t                num_owned;
};
template < el_o_t... orders, CondensationPolicy CP >
auto getCondensedNodes(const mesh::MeshPartition< orders... >& mesh,
                       const NodeCondensationMap< CP >&        cond_map) -> CondensedNodes
{
    auto retval              = CondensedNodes{};
    auto& [nodes, num_owned] = retval;
    auto owned_nodes_view    = cond_map.getCondensedOwnedNodesView(mesh) | std::views::common;
    nodes                    = std::vector< n_id_t >{owned_nodes_view.begin(), owned_nodes_view.end()};
    std::ranges::sort(nodes);
    num_owned = nodes.size();
    std::ranges::copy(cond_map.getCondensedGhostNodesView(mesh), std::back_inserter(nodes));
    std::ranges::sort(nodes | std::views::drop(num_owned));
    return retval;
}

template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
auto makeLocalDofBmp(const mesh::MeshPartition< orders... >& mesh,
                     const NodeCondensationMap< CP >&        cond_map,
                     const CondensedNodes&                   cond_nodes,
                     util::ConstexprValue< problem_def >) -> Kokkos::View< char**, Kokkos::LayoutLeft >
{
    const auto& [nodes, num_owned] = cond_nodes;
    const auto num_all             = nodes.size();
    const auto g2l                 = util::IndexMap{nodes};
    auto       retval = Kokkos::View< char**, Kokkos::LayoutLeft >{"DOF bmp", num_all, problem_def.n_fields};
    for (const auto& [domain, active_bmp] : problem_def)
    {
        const auto active_inds = util::getTrueInds(active_bmp);
        const auto mark_node   = [&](n_id_t node) {
            const auto cond_gid = cond_map.getCondensedId(node);
            const auto cond_lid = g2l(cond_gid);
            for (auto j : active_inds)
                std::atomic_ref{retval(cond_lid, j)}.store(true, std::memory_order_relaxed);
        };
        const auto mark_el = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& el) {
            for (auto n : getPrimaryNodesView< CP >(el))
                mark_node(n);
        };
        mesh.visit(mark_el, domain, std::execution::par);
    }
    return retval;
}

inline auto makeCondensedCommContext(const MpiComm&        comm,
                                     const CondensedNodes& cond) -> std::shared_ptr< const comm::ImportExportContext<> >
{
    const auto& [nodes, num_owned] = cond;
    const auto nodes_signed        = util::ArrayOwner< global_dof_t >{nodes};
    const auto owned               = std::span{nodes_signed | std::views::take(num_owned)};
    const auto shared              = std::span{nodes_signed | std::views::drop(num_owned)};
    return std::make_shared< const comm::ImportExportContext<> >(comm, owned, shared);
}

inline auto doCommsAndCountOwnedDofs(const MpiComm&                                              comm,
                                     const std::shared_ptr< const comm::ImportExportContext<> >& context,
                                     const Kokkos::View< char**, Kokkos::LayoutLeft >&           dof_bmp,
                                     size_t                                                      num_owned) -> size_t
{
    using namespace std::views;
    const auto num_all       = dof_bmp.extent(0);
    const auto dofs_per_node = dof_bmp.extent(1);
    auto       importer      = comm::Import< char, local_dof_t >{context, dofs_per_node};
    auto       exporter      = comm::Export< char, local_dof_t >{context, dofs_per_node};
    const auto shared_range  = std::make_pair(num_owned, num_all);
    const auto shared_view   = Kokkos::subview(dof_bmp, shared_range, Kokkos::ALL());
    importer.setOwned(dof_bmp);
    importer.setShared(shared_view);
    exporter.setOwned(dof_bmp);
    exporter.setShared(shared_view);
    exporter.doBlockingExport(comm, util::AtomicOrInto{});
    importer.postComms(comm);
    // We can count the number of owned DOFs while we wait for importing to complete
    const auto dofs_flat  = util::flatten(dof_bmp);
    const auto take_owned = [num_owned](auto&& v) {
        return std::forward< decltype(v) >(v) | take(num_owned);
    };
    auto       owned_view_flat = dofs_flat | chunk(num_all) | transform(take_owned) | join;
    const auto retval          = std::ranges::count(owned_view_flat, true);
    importer.wait();
    return static_cast< size_t >(retval);
}

inline auto computeBaseDof(const MpiComm& comm, size_t num_owned_dofs) -> global_dof_t
{
    auto scan_view = std::views::single(num_owned_dofs);
    comm.exclusiveScanInPlace(scan_view, MPI_SUM);
    return static_cast< global_dof_t >(comm.getRank() == 0 ? 0 : scan_view.front());
}

inline auto computeOwnedDofs(const Kokkos::View< char**, Kokkos::LayoutLeft >& dof_bmp,
                             size_t                                            num_owned_nodes,
                             global_dof_t base_dof) -> Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >
{
    const auto dofs_per_node = dof_bmp.extent(1);
    auto       retval        = Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >{"DOFs", dof_bmp.layout()};
    for (size_t i = 0; i != num_owned_nodes; ++i)
        for (size_t j = 0; j != dofs_per_node; ++j)
            retval(i, j) = dof_bmp(i, j) ? base_dof++ : NodeToGlobalDofMap< 1 >::invalid_dof;
    return retval;
}

inline void communicateSharedDofs(const MpiComm&                                              comm,
                                  const std::shared_ptr< const comm::ImportExportContext<> >& context,
                                  const Kokkos::View< global_dof_t**, Kokkos::LayoutLeft >&   dofs,
                                  size_t                                                      num_owned)
{
    const auto num_all       = dofs.extent(0);
    const auto dofs_per_node = dofs.extent(1);
    auto       importer      = comm::Import< global_dof_t, local_dof_t >{context, dofs_per_node};
    const auto shared_range  = std::make_pair(num_owned, num_all);
    const auto shared_view   = Kokkos::subview(dofs, shared_range, Kokkos::ALL());
    importer.setOwned(dofs);
    importer.setShared(shared_view);
    importer.doBlockingImport(comm);
}
} // namespace detail

template < size_t dofs_per_node >
void NodeToGlobalDofMap< dofs_per_node >::initFromDofs(
    std::span< const n_id_t >                                       cond_nodes,
    size_t                                                          num_owned,
    const Kokkos::View< const global_dof_t**, Kokkos::LayoutLeft >& dofs)
{
    m_map = map_t(cond_nodes.size());
    for (auto&& [i, cond_node] : cond_nodes | std::views::enumerate)
    {
        auto node_dofs = payload_t{};
        for (auto&& [j, nd] : node_dofs | std::views::enumerate)
            nd = dofs(i, j);
        m_map[cond_node] = node_dofs;
        if (static_cast< size_t >(i) < num_owned)
            m_num_owned += std::ranges::count_if(node_dofs, &isValid);
    }
}

template < size_t dofs_per_node >
template < el_o_t... orders, CondensationPolicy CP, ProblemDef problem_def >
NodeToGlobalDofMap< dofs_per_node >::NodeToGlobalDofMap(const MpiComm&                          comm,
                                                        const mesh::MeshPartition< orders... >& mesh,
                                                        const NodeCondensationMap< CP >&        cond_map,
                                                        util::ConstexprValue< problem_def >     problemdef_ctwrpr)
    requires(problem_def.n_fields == dofs_per_node)
{
    L3STER_PROFILE_FUNCTION;
    const auto cond_node_info     = detail::getCondensedNodes(mesh, cond_map);
    const auto [nodes, num_owned] = cond_node_info;
    const auto dof_bmp            = detail::makeLocalDofBmp(mesh, cond_map, cond_node_info, problemdef_ctwrpr);
    const auto context            = detail::makeCondensedCommContext(comm, cond_node_info);
    const auto num_owned_dofs     = detail::doCommsAndCountOwnedDofs(comm, context, dof_bmp, num_owned);
    const auto base_dof           = detail::computeBaseDof(comm, num_owned_dofs);
    const auto dofs               = detail::computeOwnedDofs(dof_bmp, num_owned, base_dof);
    detail::communicateSharedDofs(comm, context, dofs, num_owned);
    initFromDofs(nodes, num_owned, dofs);
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_NODETODOFMAP_HPP
