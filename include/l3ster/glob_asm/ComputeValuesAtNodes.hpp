#ifndef L3STER_GLOB_ASM_COMPUTEVALUESATNODES_HPP
#define L3STER_GLOB_ASM_COMPUTEVALUESATNODES_HPP

#include "l3ster/basisfun/ReferenceBasisAtNodes.hpp"
#include "l3ster/glob_asm/AssembleGlobalSystem.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/util/Ranges.hpp"

#include <atomic>

namespace lstr::glob_asm
{
namespace detail
{
template < size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs, size_t num_maps >
auto getNodeDofsAtInds(const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& map,
                       const std::array< dofind_t, n_dofs >&                         dof_inds,
                       n_id_t                                                        node)
{
    return util::makeIndexedView(map(node).front(), dof_inds);
}

template < size_t n_rhs >
void prepValsAndParentsAtDofs(auto&&                                         dof_range,
                              std::vector< std::uint8_t >&                   parents,
                              const std::array< std::span< val_t >, n_rhs >& values)
{
    for (auto dof : std::forward< decltype(dof_range) >(dof_range))
    {
        std::atomic_ref{parents[dof]}.fetch_add(1, std::memory_order_relaxed);
        for (size_t i = 0; i != n_rhs; ++i)
            std::atomic_ref{values[i][dof]}.store(0., std::memory_order_relaxed);
    }
}

template < el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        n_dofs,
           size_t        num_maps,
           size_t        n_fields,
           size_t        n_rhs >
auto initValsAndParentsDomain(const mesh::MeshPartition< orders... >&                       mesh,
                              const util::ArrayOwner< d_id_t >&                             domain_ids,
                              const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                              const std::array< dofind_t, n_dofs >&                         dof_inds,
                              const SolutionManager::FieldValueGetter< n_fields >&,
                              const std::array< std::span< val_t >, n_rhs >& values) -> std::vector< std::uint8_t >
{
    if constexpr (n_fields != 0)
    {
        auto       retval          = std::vector< std::uint8_t >(values.front().size(), 0);
        const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            prepValsAndParentsAtDofs(
                element.getNodes() | std::views::filter(util::negatePredicate(mesh.getGhostNodePredicate())) |
                    std::views::transform([&](n_id_t node) { return getNodeDofsAtInds(dof_map, dof_inds, node); }) |
                    std::views::join,
                retval,
                values);
        };
        mesh.visit(process_element, domain_ids, std::execution::par);
        return retval;
    }
    else
        return {};
}

template < el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        n_dofs,
           size_t        num_maps,
           size_t        n_fields,
           size_t        n_rhs >
auto initValsAndParentsBoundary(const mesh::MeshPartition< orders... >&                       mesh,
                                const util::ArrayOwner< d_id_t >&                             boundary_ids,
                                const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                                const std::array< dofind_t, n_dofs >&                         dof_inds,
                                const SolutionManager::FieldValueGetter< n_fields >&,
                                const std::array< std::span< val_t >, n_rhs >& values) -> std::vector< std::uint8_t >
{
    auto       retval = std::vector< std::uint8_t >(values.front().size(), 0);
    const auto process_el_view =
        [&]< mesh::ElementType ET, el_o_t EO >(const mesh::BoundaryElementView< ET, EO >& el_view) {
            prepValsAndParentsAtDofs(
                el_view.getSideNodesView() | std::views::filter(util::negatePredicate(mesh.getGhostNodePredicate())) |
                    std::views::transform([&](n_id_t node) { return getNodeDofsAtInds(dof_map, dof_inds, node); }) |
                    std::views::join,
                retval,
                values);
        };
    mesh.visitBoundaries(process_el_view, boundary_ids, std::execution::par);
    return retval;
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
                          const std::array< std::span< val_t >, n_rhs >&                values_out)
{
    L3STER_PROFILE_FUNCTION;
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        for (auto node : element.getNodes() | std::views::filter(util::negatePredicate(mesh.getGhostNodePredicate())))
            for (size_t dof_ind = 0; auto dof : detail::getNodeDofsAtInds(dof_map, dof_inds, node))
            {
                for (size_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                {
                    const auto val  = values_in[rhs_ind][dof_ind];
                    auto&      dest = values_out[rhs_ind][dof];
                    std::atomic_ref{dest}.store(val, std::memory_order_relaxed);
                }
                ++dof_ind;
            }
    };
    mesh.visit(process_element, domain_ids, std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        num_maps,
           size_t        n_fields,
           size_t        n_rhs >
void computeValuesAtNodes(const ResidualDomainKernel< Kernel, params >&                 kernel,
                          const mesh::MeshPartition< orders... >&                       mesh,
                          const util::ArrayOwner< d_id_t >&                             ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, params.n_equations >&             dof_inds,
                          const SolutionManager::FieldValueGetter< n_fields >&          field_val_getter,
                          const std::array< std::span< val_t >, n_rhs >&                values,
                          val_t                                                         time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = detail::checkDomainDimension(mesh, ids, params.dimension);
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the domain");

    const auto num_parents = detail::initValsAndParentsDomain(mesh, ids, dof_map, dof_inds, field_val_getter, values);
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes       = element.getNodes();
            const auto& basis_at_nodes = basis::getBasisAtNodes< ET, EO >();
            const auto& node_locations = mesh::getNodeLocations< ET, EO >();

            const auto node_vals            = field_val_getter(el_nodes);
            const auto jacobi_mat_generator = map::getNatJacobiMatGenerator(element);

            const auto process_node = [&](size_t node_ind) {
                const auto node        = el_nodes[node_ind];
                const auto ref_coords  = node_locations[node_ind];
                const auto jacobi_mat  = jacobi_mat_generator(ref_coords);
                const auto phys_coords = map::mapToPhysicalSpace(element, ref_coords);
                const auto phys_basis_ders =
                    map::computePhysBasisDers(jacobi_mat, basis_at_nodes.derivatives[node_ind]);
                const auto field_vals = computeFieldVals(basis_at_nodes.values[node_ind], node_vals);
                const auto field_ders = computeFieldDers(phys_basis_ders, node_vals);
                const auto point      = SpaceTimePoint{phys_coords, time};
                const auto kernel_in  = typename KernelInterface< params >::DomainInput{field_vals, field_ders, point};
                const auto ker_res    = kernel(kernel_in);
                for (size_t dof_ind = 0; auto dof : detail::getNodeDofsAtInds(dof_map, dof_inds, node))
                {
                    if constexpr (n_fields != 0)
                    {
                        const auto np_fp       = static_cast< val_t >(num_parents[dof]);
                        auto       update_vals = std::array< val_t, n_rhs >{};
                        for (size_t rhs_ind = 0; val_t & upd_val : update_vals)
                            upd_val = ker_res(dof_ind, rhs_ind++) /= np_fp;
                        for (size_t rhs_ind = 0; val_t upd_val : update_vals)
                        {
                            auto& dest = values[rhs_ind++][dof];
                            std::atomic_ref{dest}.fetch_add(upd_val, std::memory_order_relaxed);
                        }
                    }
                    else
                        for (size_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                        {
                            const auto upd_val = ker_res(dof_ind, rhs_ind);
                            auto&      dest    = values[rhs_ind][dof];
                            std::atomic_ref{dest}.store(upd_val, std::memory_order_relaxed);
                        }
                    ++dof_ind;
                }
            };
            std::ranges::for_each(std::views::iota(0u, el_nodes.size()) | std::views::filter([&](auto node_ind) {
                                      return not mesh.isGhostNode(el_nodes[node_ind]);
                                  }),
                                  process_node);
        }
    };
    mesh.visit(process_element, ids, std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        num_maps,
           size_t        n_fields,
           size_t        n_rhs >
void computeValuesAtNodes(const ResidualBoundaryKernel< Kernel, params >&               kernel,
                          const mesh::MeshPartition< orders... >&                       mesh,
                          const util::ArrayOwner< d_id_t >&                             ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, params.n_equations >&             dof_inds,
                          const SolutionManager::FieldValueGetter< n_fields >&          field_val_getter,
                          const std::array< std::span< val_t >, n_rhs >&                values,
                          val_t                                                         time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = detail::checkDomainDimension(mesh, ids, params.dimension - 1);
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the boundary");

    const auto num_parents = detail::initValsAndParentsBoundary(mesh, ids, dof_map, dof_inds, field_val_getter, values);
    const auto process_el_view = [&]< mesh::ElementType ET, el_o_t EO >(mesh::BoundaryElementView< ET, EO > el_view) {
        if constexpr (params.dimension == mesh::Element< ET, EO >::native_dim)
        {
            const auto& el_nodes       = el_view->getNodes();
            const auto& basis_at_nodes = basis::getBasisAtNodes< ET, EO >();
            const auto& node_locations = mesh::getNodeLocations< ET, EO >();

            const auto node_vals            = field_val_getter(el_nodes);
            const auto jacobi_mat_generator = map::getNatJacobiMatGenerator(*el_view);

            const auto process_node = [&](size_t node_ind) {
                const auto node        = el_nodes[node_ind];
                const auto ref_coords  = node_locations[node_ind];
                const auto jacobi_mat  = jacobi_mat_generator(ref_coords);
                const auto phys_coords = map::mapToPhysicalSpace(*el_view, ref_coords);
                const auto normal      = map::computeBoundaryNormal(el_view, jacobi_mat);
                const auto phys_basis_ders =
                    map::computePhysBasisDers(jacobi_mat, basis_at_nodes.derivatives[node_ind]);
                const auto field_vals = computeFieldVals(basis_at_nodes.values[node_ind], node_vals);
                const auto field_ders = computeFieldDers(phys_basis_ders, node_vals);
                const auto point      = SpaceTimePoint{phys_coords, time};
                const auto kernel_in =
                    typename KernelInterface< params >::BoundaryInput{field_vals, field_ders, point, normal};
                const auto ker_res = kernel(kernel_in);
                for (size_t dof_ind = 0; auto dof : detail::getNodeDofsAtInds(dof_map, dof_inds, node))
                {
                    const auto np_fp       = static_cast< double >(num_parents[dof]);
                    const auto update_vals = std::invoke([&] {
                        auto retval = std::array< val_t, n_rhs >{};
                        for (size_t rhs_ind = 0; val_t & r : retval)
                            r = ker_res(dof_ind, rhs_ind++);
                        for (val_t& upd_val : retval)
                            upd_val /= np_fp;
                        return retval;
                    });
                    for (size_t rhs_ind = 0; val_t upd_val : update_vals)
                        std::atomic_ref{values[rhs_ind++][dof]}.fetch_add(upd_val, std::memory_order_relaxed);
                    ++dof_ind;
                }
            };
            std::ranges::for_each(el_view.getSideNodeInds() | std::views::filter([&](size_t node_ind) {
                                      return not mesh.isGhostNode(el_nodes[node_ind]);
                                  }),
                                  process_node);
        }
    };
    mesh.visitBoundaries(process_el_view, ids, std::execution::par);
}
} // namespace lstr::glob_asm
#endif // L3STER_GLOB_ASM_COMPUTEVALUESATNODES_HPP
