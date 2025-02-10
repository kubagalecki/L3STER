#ifndef L3STER_ALGSYS_COMPUTEVALUESATNODES_HPP
#define L3STER_ALGSYS_COMPUTEVALUESATNODES_HPP

#include "l3ster/algsys/AssembleGlobalSystem.hpp"
#include "l3ster/basisfun/ReferenceBasisAtNodes.hpp"
#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/util/Ranges.hpp"

#include <atomic>

namespace lstr::algsys
{
/// Access separate owned/shared views using linear index
/// Views types are erased so that the accessor is trivial
template < Arithmetic_c T >
class BorderAccessor
{
public:
    template < util::KokkosView_c View >
    BorderAccessor(const View& owned, const View& shared, local_dof_t num_owned)
        requires(View::rank() == 2)
        : m_owned{owned.data()},
          m_shared{shared.data()},
          m_num_owned{num_owned},
          m_owned_stride{static_cast< local_dof_t >(owned.stride(1))},
          m_shared_stride{static_cast< local_dof_t >(shared.stride(1))}
    {}
    template < util::KokkosView_c View >
    BorderAccessor(const View& owned, const View& shared)
        requires(View::rank() == 2)
        : BorderAccessor(owned, shared, static_cast< local_dof_t >(owned.extent(0)))
    {}

    decltype(auto) operator()(local_dof_t row, local_dof_t col) const
    {
        const auto is_owned = row < m_num_owned;
        const auto offset   = is_owned * row + not is_owned * (row - m_num_owned);
        return is_owned ? m_owned[col * m_owned_stride + offset] : m_shared[col * m_shared_stride + offset];
    }

private:
    T*          m_owned;
    T*          m_shared;
    local_dof_t m_num_owned, m_owned_stride, m_shared_stride;
};
template < util::KokkosView_c View >
BorderAccessor(const View&, const View&) -> BorderAccessor< typename View::value_type >;
template < util::KokkosView_c View >
BorderAccessor(const View&, const View&, local_dof_t) -> BorderAccessor< typename View::value_type >;

/// Just a trivial 2D view
template < Arithmetic_c T >
class InteriorAccessor
{
public:
    template < util::KokkosView_c View >
    InteriorAccessor(const View& view) : m_data{view.data()}, m_stride{static_cast< local_dof_t >(view.stride(1))}
    {}

    decltype(auto) operator()(local_dof_t row, local_dof_t col) const { return m_data[col * m_stride + row]; }

private:
    T*          m_data;
    local_dof_t m_stride;
};
template < util::KokkosView_c View >
InteriorAccessor(const View& view) -> InteriorAccessor< typename View::value_type >;

namespace detail
{
template < size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs, size_t num_maps >
auto getNodeDofsAtInds(const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& map,
                       const std::array< dofind_t, n_dofs >&                         dof_inds,
                       n_id_t                                                        node)
{
    return util::makeIndexedView(map(node).back(), dof_inds);
}

template < size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs >
auto getNodeDofsAtInds(const dofs::LocalDofMap< max_dofs_per_node >& map,
                       const std::array< dofind_t, n_dofs >&         dof_inds,
                       n_loc_id_t                                    node)
{
    return util::makeIndexedView(map(node), dof_inds);
}

template < el_o_t... orders, size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs, size_t num_maps >
void zeroOut(const mesh::MeshPartition< orders... >&                       mesh,
             const util::ArrayOwner< d_id_t >&                             domain_ids,
             const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
             const std::array< dofind_t, n_dofs >&                         dof_inds,
             const tpetra_multivector_t::host_view_type&                   values_out)
{
    const auto num_rhs         = values_out.extent(1);
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        for (auto node : element.getNodes())
            for (auto&& [dof_ind, dof] : detail::getNodeDofsAtInds(dof_map, dof_inds, node) | std::views::enumerate)
                for (size_t rhs = 0; rhs != num_rhs; ++rhs)
                {
                    auto& dest = values_out(dof, rhs);
                    std::atomic_ref{dest}.store(0., std::memory_order_relaxed);
                }
    };
    mesh.visit(process_element, domain_ids, std::execution::par);
}

template < std::invocable DoInterior, std::invocable DoBorder >
void averageElementContributions(const MpiComm&                              comm,
                                 const tpetra_multivector_t::host_view_type& owned_values,
                                 const tpetra_multivector_t::host_view_type& shared_values,
                                 std::span< std::uint16_t >                  num_contribs,
                                 comm::Export< val_t, local_dof_t >&         exporter,
                                 DoInterior&&                                do_interior,
                                 DoBorder&&                                  do_border)
{
    const size_t num_owned  = owned_values.extent(0);
    const size_t num_shared = shared_values.extent(0);
    const size_t num_rhs    = owned_values.extent(1);

    exporter.setOwned(owned_values);
    exporter.setShared(shared_values);

    auto contrib_export = comm::Export< std::uint16_t, local_dof_t >{exporter.getContext(), 1};
    contrib_export.setOwned(num_contribs, num_owned);
    contrib_export.setShared(std::span{num_contribs}.subspan(num_owned), num_shared);
    contrib_export.postRecvs(comm);
    contrib_export.postSends(comm);

    exporter.postRecvs(comm);
    std::invoke(std::forward< DoBorder >(do_border));
    exporter.postSends(comm);
    std::invoke(std::forward< DoInterior >(do_interior));

    // We need to zero out owned entries which are written to *only* by neighbors
    auto owned_contributed = util::DynamicBitset(num_owned);
    for (auto&& [i, nc] : num_contribs | std::views::take(num_owned) | std::views::enumerate)
        owned_contributed.assign(static_cast< size_t >(i), nc > 0);
    contrib_export.wait(util::AtomicSumInto{});
    for (auto&& [i, nc] : num_contribs | std::views::take(num_owned) | std::views::enumerate)
        if (num_contribs[static_cast< size_t >(i)] > 0 and not owned_contributed.test(static_cast< size_t >(i)))
            for (size_t rhs = 0; rhs != num_rhs; ++rhs)
                owned_values(i, rhs) = 0.;
    exporter.wait(util::AtomicSumInto{});

    for (auto&& [i, nc] : num_contribs | std::views::take(num_owned) | std::views::enumerate)
        if (nc) // nc == 0 means nobody wrote to this index, so we keep the original value
            for (size_t rhs = 0; rhs != num_rhs; ++rhs)
                owned_values(i, rhs) /= static_cast< val_t >(nc);
}
} // namespace detail

template < el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        n_dofs,
           size_t        num_maps,
           size_t        n_rhs >
void computeValuesAtNodes(const mesh::MeshPartition< orders... >&                       mesh,
                          const util::ArrayOwner< d_id_t >&                             domain_ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, n_dofs >&                         dof_inds,
                          const std::array< std::span< const val_t, n_dofs >, n_rhs >&  values_in,
                          const tpetra_multivector_t::host_view_type&                   values_out,
                          const tpetra_multivector_t::host_view_type&                   num_contribs)
{
    L3STER_PROFILE_FUNCTION;
    util::throwingAssert(values_out.extent(1) == n_rhs);
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        for (auto node : element.getNodes())
            for (auto&& [dof_ind, dof] : detail::getNodeDofsAtInds(dof_map, dof_inds, node) | std::views::enumerate)
            {
                std::atomic_ref{num_contribs(dof, 0)}.store(1., std::memory_order_relaxed);
                for (size_t rhs = 0; rhs != n_rhs; ++rhs)
                {
                    const auto value = values_in[rhs][dof_ind];
                    auto&      dest  = values_out(dof, rhs);
                    std::atomic_ref{dest}.store(value, std::memory_order_relaxed);
                }
            }
    };
    mesh.visit(process_element, domain_ids, std::execution::par);
}

template < el_o_t... orders, size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs, size_t n_rhs >
void computeValuesAtNodes(const MpiComm&                                               comm,
                          const mesh::LocalMeshView< orders... >&                      interior_mesh,
                          const mesh::LocalMeshView< orders... >&                      border_mesh,
                          comm::Export< val_t, local_dof_t >&                          exporter,
                          const util::ArrayOwner< d_id_t >&                            domain_ids,
                          const dofs::LocalDofMap< max_dofs_per_node >&                dof_map,
                          const std::array< dofind_t, n_dofs >&                        dof_inds,
                          const std::array< std::span< const val_t, n_dofs >, n_rhs >& values_in,
                          const tpetra_multivector_t::host_view_type&                  owned_values)
{
    L3STER_PROFILE_FUNCTION;
    const auto max_dim     = std::max(interior_mesh.getMaxDim(), border_mesh.getMaxDim());
    const auto has_max_dim = [&](d_id_t domain) {
        return max_dim == interior_mesh.getDomainDim(domain).value_or(max_dim) or
               max_dim == border_mesh.getDomainDim(domain).value_or(max_dim);
    };
    const auto not_max_dim = [&](d_id_t domain) {
        return max_dim != interior_mesh.getDomainDim(domain).value_or(invalid_dim) and
               max_dim != border_mesh.getDomainDim(domain).value_or(invalid_dim);
    };
    const bool all_max_dim  = std::ranges::all_of(domain_ids, has_max_dim);
    const auto none_max_dim = std::ranges::all_of(domain_ids, not_max_dim);
    util::throwingAssert(all_max_dim xor none_max_dim, "Cannot mix domain and boundary IDs");

    const auto num_owned_dofs  = static_cast< local_dof_t >(dof_map.getNumOwnedDofs());
    const auto num_shared_dofs = dof_map.getNumSharedDofs();
    auto       num_contribs    = util::ArrayOwner< std::uint16_t >(dof_map.getNumTotalDofs(), 0);
    auto       shared_values   = tpetra_multivector_t::host_view_type("shared vals", num_shared_dofs, n_rhs);
    const auto border_accessor = BorderAccessor{owned_values, shared_values, num_owned_dofs};
    const auto set_node_dofs   = [&](n_loc_id_t node) {
        const auto& node_dofs = dof_map(node);
        for (auto&& [dof_ind, dof] : util::makeIndexedView(node_dofs, dof_inds) | std::views::enumerate)
        {
            std::atomic_ref{num_contribs.at(dof)}.store(1, std::memory_order_relaxed);
            for (size_t rhs = 0; rhs != n_rhs; ++rhs)
            {
                const auto value = values_in[rhs][dof_ind];
                auto&      dest  = border_accessor(dof, rhs);
                std::atomic_ref{dest}.store(value, std::memory_order_relaxed);
            }
        }
    };
    const auto do_mesh = [&](const mesh::LocalMeshView< orders... >& mesh) {
        if (all_max_dim)
        {
            const auto do_element =
                [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
                    const auto& nodes = element.getLocalNodes();
                    std::ranges::for_each(nodes, set_node_dofs);
                };
            mesh.visit(do_element, domain_ids, std::execution::par);
        }
        else
        {
            const auto do_element_boundary =
                [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementBoundaryView< ET, EO >& el_view) {
                    const auto& nodes = el_view->getLocalNodes();
                    std::ranges::for_each(util::makeIndexedView(nodes, el_view.getSideNodeInds()), set_node_dofs);
                };
            mesh.visitBoundaries(do_element_boundary, domain_ids, std::execution::par);
        }
    };
    const auto do_interior = [&] {
        do_mesh(interior_mesh);
    };
    const auto do_border = [&] {
        do_mesh(border_mesh);
    };
    detail::averageElementContributions(
        comm, owned_values, shared_values, num_contribs, exporter, do_interior, do_border);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        num_maps,
           size_t        n_fields >
void computeValuesAtNodes(const ResidualDomainKernel< Kernel, params >&                 kernel,
                          const mesh::MeshPartition< orders... >&                       mesh,
                          const util::ArrayOwner< d_id_t >&                             domain_ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, params.n_equations >&             dof_inds,
                          const post::FieldAccess< n_fields >&                          field_val_getter,
                          const tpetra_multivector_t::host_view_type&                   values,
                          const tpetra_multivector_t::host_view_type&                   num_contribs,
                          val_t                                                         time)
{
    L3STER_PROFILE_FUNCTION;
    const bool dim_match = detail::checkDomainDimension(mesh, domain_ids, params.dimension);
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the domain");
    util::throwingAssert(params.n_rhs == values.extent(1));
    detail::zeroOut(mesh, domain_ids, dof_map, dof_inds, values);
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes       = element.getNodes();
            const auto& el_data        = element.getData();
            const auto& basis_at_nodes = basis::getBasisAtNodes< ET, EO >();
            const auto& node_locations = mesh::getNodeLocations< ET, EO >();
            const auto  node_vals      = field_val_getter.getGloballyIndexed(el_nodes);
            const auto  jacobi_gen     = map::getNatJacobiMatGenerator(el_data);
            const auto  process_node   = [&](size_t node_ind) {
                const auto& ref_coords    = node_locations[node_ind];
                const auto& ref_val       = basis_at_nodes.values[node_ind];
                const auto& ref_ders      = basis_at_nodes.derivatives[node_ind];
                const auto [phys_ders, _] = map::mapDomain< ET, EO >(jacobi_gen, ref_coords, ref_ders);
                const auto node           = el_nodes[node_ind];
                const auto ker_res = evalKernel(kernel, ref_coords, ref_val, phys_ders, node_vals, el_data, time);
                for (auto&& [dof_ind, dof] : detail::getNodeDofsAtInds(dof_map, dof_inds, node) | std::views::enumerate)
                {
                    std::atomic_ref{num_contribs(dof, 0)}.fetch_add(1., std::memory_order_relaxed);
                    for (size_t rhs = 0; rhs != params.n_rhs; ++rhs)
                    {
                        const auto value = ker_res(dof_ind, rhs);
                        auto&      dest  = values(dof, rhs);
                        std::atomic_ref{dest}.fetch_add(value, std::memory_order_relaxed);
                    }
                }
            };
            std::ranges::for_each(std::views::iota(0u, el_nodes.size()), process_node);
        }
    };
    mesh.visit(process_element, domain_ids, std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        n_fields >
void computeValuesAtNodes(const ResidualDomainKernel< Kernel, params >&     kernel,
                          const MpiComm&                                    comm,
                          const mesh::LocalMeshView< orders... >&           mesh_interior,
                          const mesh::LocalMeshView< orders... >&           mesh_border,
                          comm::Export< val_t, local_dof_t >&               exporter,
                          const util::ArrayOwner< d_id_t >&                 domain_ids,
                          const dofs::LocalDofMap< max_dofs_per_node >&     dof_map,
                          const std::array< dofind_t, params.n_equations >& dof_inds,
                          const post::FieldAccess< n_fields >&              field_val_getter,
                          const tpetra_multivector_t::host_view_type&       owned_values,
                          val_t                                             time)
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto n_rhs = params.n_rhs;
    util::throwingAssert(mesh_interior.checkDomainDims(domain_ids, params.dimension));
    util::throwingAssert(mesh_border.checkDomainDims(domain_ids, params.dimension));
    util::throwingAssert(owned_values.extent(1) == n_rhs);
    const auto num_owned_dofs  = static_cast< local_dof_t >(dof_map.getNumOwnedDofs());
    const auto num_shared_dofs = dof_map.getNumSharedDofs();
    auto       num_contribs    = util::ArrayOwner< std::uint16_t >(dof_map.getNumTotalDofs(), 0);
    auto       shared_values   = tpetra_multivector_t::host_view_type("shared vals", num_shared_dofs, n_rhs);
    const auto border_accessor = BorderAccessor{owned_values, shared_values, num_owned_dofs};
    const auto init = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes = element.getLocalNodes();
            for (auto node : el_nodes)
                for (auto dof : detail::getNodeDofsAtInds(dof_map, dof_inds, node))
                {
                    for (local_dof_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                        std::atomic_ref{border_accessor(dof, rhs_ind)}.store(0., std::memory_order_relaxed);
                    std::atomic_ref{num_contribs.at(dof)}.fetch_add(1, std::memory_order_relaxed);
                }
        }
    };
    const auto visitor = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementView< ET, EO >& element) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes       = element.getLocalNodes();
            const auto& basis_at_nodes = basis::getBasisAtNodes< ET, EO >();
            const auto& node_locations = mesh::getNodeLocations< ET, EO >();
            const auto  node_vals      = field_val_getter.getLocallyIndexed(el_nodes);
            const auto  jacobi_gen     = map::getNatJacobiMatGenerator(element.getData());
            for (auto&& [node_ind, node] : el_nodes | std::views::enumerate)
            {
                const auto& ref_coords    = node_locations[node_ind];
                const auto& ref_ders      = basis_at_nodes.derivatives[node_ind];
                const auto [phys_ders, _] = map::mapDomain< ET, EO >(jacobi_gen, ref_coords, ref_ders);
                const auto ker_res        = evalKernel(
                    kernel, ref_coords, basis_at_nodes.values[node_ind], phys_ders, node_vals, element.getData(), time);
                for (auto&& [dof_ind, dof] : detail::getNodeDofsAtInds(dof_map, dof_inds, node) | std::views::enumerate)
                    for (size_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                    {
                        const val_t increment = ker_res(dof_ind, rhs_ind);
                        val_t&      dest      = border_accessor(dof, static_cast< local_dof_t >(rhs_ind));
                        std::atomic_ref{dest}.fetch_add(increment, std::memory_order_relaxed);
                    }
            }
        }
    };
    const auto do_interior = [&] {
        mesh_interior.visit(visitor, domain_ids, std::execution::par);
    };
    const auto do_border = [&] {
        mesh_border.visit(visitor, domain_ids, std::execution::par);
    };

    mesh_interior.visit(init, domain_ids, std::execution::par);
    mesh_border.visit(init, domain_ids, std::execution::par);
    detail::averageElementContributions(
        comm, owned_values, shared_values, num_contribs, exporter, do_interior, do_border);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        num_maps,
           size_t        n_fields >
void computeValuesAtNodes(const ResidualBoundaryKernel< Kernel, params >&               kernel,
                          const mesh::MeshPartition< orders... >&                       mesh,
                          const util::ArrayOwner< d_id_t >&                             domain_ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, params.n_equations >&             dof_inds,
                          const post::FieldAccess< n_fields >&                          field_val_getter,
                          const tpetra_multivector_t::host_view_type&                   values,
                          const tpetra_multivector_t::host_view_type&                   num_contribs,
                          val_t                                                         time)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = detail::checkDomainDimension(mesh, domain_ids, params.dimension - 1);
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the boundary");
    util::throwingAssert(params.n_rhs == values.extent(1));
    detail::zeroOut(mesh, domain_ids, dof_map, dof_inds, values);
    const auto process_el_view = [&]< mesh::ElementType ET, el_o_t EO >(mesh::BoundaryElementView< ET, EO > el_view) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes       = el_view->getNodes();
            const auto& basis_at_nodes = basis::getBasisAtNodes< ET, EO >();
            const auto& node_locations = mesh::getNodeLocations< ET, EO >();
            const auto  node_vals      = field_val_getter.getGloballyIndexed(el_nodes);
            const auto& el_data        = el_view->getData();
            const auto  jacobi_gen     = map::getNatJacobiMatGenerator(el_data);
            const auto  side           = el_view.getSide();
            const auto  process_node   = [&](size_t node_ind) {
                const auto  node                  = el_nodes[node_ind];
                const auto& ref_coords            = node_locations[node_ind];
                const auto& ref_ders              = basis_at_nodes.derivatives[node_ind];
                const auto& basis_val             = basis_at_nodes.values[node_ind];
                const auto [phys_ders, _, normal] = map::mapBoundary< ET, EO >(jacobi_gen, ref_coords, ref_ders, side);
                const auto ker_res =
                    evalKernel(kernel, ref_coords, basis_val, phys_ders, node_vals, el_data, time, normal);
                for (auto&& [dof_ind, dof] : detail::getNodeDofsAtInds(dof_map, dof_inds, node) | std::views::enumerate)
                {
                    std::atomic_ref{num_contribs(dof, 0)}.fetch_add(1., std::memory_order_relaxed);
                    for (size_t rhs = 0; rhs != params.n_rhs; ++rhs)
                    {
                        const auto value = ker_res(dof_ind, rhs);
                        auto&      dest  = values(dof, rhs);
                        std::atomic_ref{dest}.fetch_add(value, std::memory_order_relaxed);
                    }
                }
            };
            std::ranges::for_each(el_view.getSideNodeInds(), process_node);
        }
    };
    mesh.visitBoundaries(process_el_view, domain_ids, std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        n_fields >
void computeValuesAtNodes(const ResidualBoundaryKernel< Kernel, params >&   kernel,
                          const MpiComm&                                    comm,
                          const mesh::LocalMeshView< orders... >&           mesh_interior,
                          const mesh::LocalMeshView< orders... >&           mesh_border,
                          comm::Export< val_t, local_dof_t >&               exporter,
                          const util::ArrayOwner< d_id_t >&                 boundary_ids,
                          const dofs::LocalDofMap< max_dofs_per_node >&     dof_map,
                          const std::array< dofind_t, params.n_equations >& dof_inds,
                          const post::FieldAccess< n_fields >&              field_val_getter,
                          const tpetra_multivector_t::host_view_type&       owned_values,
                          val_t                                             time)
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto n_rhs = params.n_rhs;
    util::throwingAssert(mesh_interior.checkDomainDims(boundary_ids, params.dimension - 1));
    util::throwingAssert(mesh_border.checkDomainDims(boundary_ids, params.dimension - 1));
    util::throwingAssert(owned_values.extent(1) == n_rhs);
    const auto num_owned_dofs  = static_cast< local_dof_t >(dof_map.getNumOwnedDofs());
    const auto num_shared_dofs = dof_map.getNumSharedDofs();
    auto       num_contribs    = util::ArrayOwner< std::uint16_t >(dof_map.getNumTotalDofs(), 0);
    auto       shared_values   = tpetra_multivector_t::host_view_type("shared vals", num_shared_dofs, n_rhs);
    const auto border_accessor = BorderAccessor{owned_values, shared_values, num_owned_dofs};
    const auto init = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::LocalElementBoundaryView< ET, EO >& el_view) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes = el_view->getLocalNodes();
            for (auto node_ind : el_view.getSideNodeInds())
            {
                const auto node = el_nodes[node_ind];
                for (auto dof : detail::getNodeDofsAtInds(dof_map, dof_inds, node))
                {
                    for (local_dof_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                        std::atomic_ref{border_accessor(dof, rhs_ind)}.store(0., std::memory_order_relaxed);
                    std::atomic_ref{num_contribs.at(dof)}.fetch_add(1, std::memory_order_relaxed);
                }
            }
        }
    };
    const auto visitor = [&]< mesh::ElementType ET, el_o_t EO >(
                             const mesh::LocalElementBoundaryView< ET, EO >& el_view) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes       = el_view->getLocalNodes();
            const auto& basis_at_nodes = basis::getBasisAtNodes< ET, EO >();
            const auto& node_locations = mesh::getNodeLocations< ET, EO >();
            const auto  node_vals      = field_val_getter.getLocallyIndexed(el_nodes);
            const auto& el_data        = el_view->getData();
            const auto  jacobi_gen     = map::getNatJacobiMatGenerator(el_data);
            const auto  side           = el_view.getSide();
            for (auto node_ind : el_view.getSideNodeInds())
            {
                const auto  node                  = el_nodes[node_ind];
                const auto& ref_coords            = node_locations[node_ind];
                const auto& ref_ders              = basis_at_nodes.derivatives[node_ind];
                const auto& basis_vals            = basis_at_nodes.values[node_ind];
                const auto [phys_ders, _, normal] = map::mapBoundary< ET, EO >(jacobi_gen, ref_coords, ref_ders, side);
                const auto ker_res =
                    evalKernel(kernel, ref_coords, basis_vals, phys_ders, node_vals, el_data, time, normal);
                for (auto&& [dof_ind, dof] : detail::getNodeDofsAtInds(dof_map, dof_inds, node) | std::views::enumerate)
                    for (local_dof_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                    {
                        const val_t increment = ker_res(dof_ind, rhs_ind);
                        val_t&      dest      = border_accessor(dof, rhs_ind);
                        std::atomic_ref{dest}.fetch_add(increment, std::memory_order_relaxed);
                    }
            }
        }
    };
    const auto do_interior = [&] {
        mesh_interior.visitBoundaries(visitor, boundary_ids, std::execution::par);
    };
    const auto do_border = [&] {
        mesh_border.visitBoundaries(visitor, boundary_ids, std::execution::par);
    };

    mesh_interior.visitBoundaries(init, boundary_ids, std::execution::par);
    mesh_border.visitBoundaries(init, boundary_ids, std::execution::par);
    detail::averageElementContributions(
        comm, owned_values, shared_values, num_contribs, exporter, do_interior, do_border);
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_COMPUTEVALUESATNODES_HPP
