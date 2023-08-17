#ifndef L3STER_GLOB_ASM_COMPUTEVALUESATNODES_HPP
#define L3STER_GLOB_ASM_COMPUTEVALUESATNODES_HPP

#include "l3ster/basisfun/ReferenceBasisAtNodes.hpp"
#include "l3ster/glob_asm/AssembleGlobalSystem.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"
#include "l3ster/util/Ranges.hpp"

#include <atomic>

namespace lstr::glob_asm
{
template < size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs, size_t num_maps >
auto getNodeDofsAtInds(const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& map,
                       const std::array< dofind_t, n_dofs >&                         dof_inds,
                       n_id_t                                                        node)
{
    return util::makeIndexedView(map(node).front(), dof_inds);
}

template < el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        n_dofs,
           size_t        num_maps,
           size_t        n_fields >
auto initValsAndParents(const mesh::MeshPartition< orders... >&                       mesh,
                        mesh::DomainIdRange_c auto&&                                  domain_ids,
                        const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                        const std::array< dofind_t, n_dofs >&                         dof_inds,
                        const SolutionManager::FieldValueGetter< n_fields >&,
                        std::span< val_t > values) -> std::vector< std::uint8_t >
{
    if constexpr (n_fields != 0)
    {
        auto       retval          = std::vector< std::uint8_t >(values.size(), 0);
        const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            for (auto node :
                 element.getNodes() | std::views::filter([&](auto node) { return not mesh.isGhostNode(node); }))
            {
                for (auto dof : getNodeDofsAtInds(dof_map, dof_inds, node))
                {
                    std::atomic_ref{retval[dof]}.fetch_add(1, std::memory_order_relaxed);
                    std::atomic_ref{values[dof]}.store(0., std::memory_order_relaxed);
                }
            }
        };
        mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
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
           size_t        n_fields >
auto initValsAndParents(const mesh::BoundaryView< orders... >&                        boundary,
                        const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                        const std::array< dofind_t, n_dofs >&                         dof_inds,
                        const SolutionManager::FieldValueGetter< n_fields >&,
                        std::span< val_t > values) -> std::vector< std::uint8_t >
{
    auto       retval = std::vector< std::uint8_t >(values.size(), 0);
    const auto process_element =
        [&]< mesh::ElementType ET, el_o_t EO >(const mesh::BoundaryElementView< ET, EO >& el_view) {
            const auto process_node = [&](n_id_t node) {
                std::ranges::for_each(getNodeDofsAtInds(dof_map, dof_inds, node), [&](local_dof_t dof) {
                    std::atomic_ref{retval[dof]}.fetch_add(1, std::memory_order_relaxed);
                    std::atomic_ref{values[dof]}.store(0., std::memory_order_relaxed);
                });
            };
            std::ranges::for_each(el_view.getSideNodesView() | std::views::filter([&](n_id_t node) {
                                      return not boundary.getParent()->isGhostNode(node);
                                  }),
                                  process_node);
        };
    boundary.visit(process_element, std::execution::par);
    return retval;
}

template < el_o_t... orders, size_t max_dofs_per_node, std::integral dofind_t, size_t n_dofs, size_t num_maps >
void computeValuesAtNodes(const mesh::MeshPartition< orders... >&                       mesh,
                          mesh::DomainIdRange_c auto&&                                  domain_ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, n_dofs >&                         dof_inds,
                          std::span< const val_t, n_dofs >                              values_in,
                          std::span< val_t >                                            values_out)
{
    L3STER_PROFILE_FUNCTION;
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        const auto process_node = [&](size_t node) {
            for (size_t dof_ind = 0; auto dof : getNodeDofsAtInds(dof_map, dof_inds, node))
                std::atomic_ref{values_out[dof]}.store(values_in[dof_ind++], std::memory_order_relaxed);
        };
        std::ranges::for_each(element.getNodes() |
                                  std::views::filter([&](auto node) { return not mesh.isGhostNode(node); }),
                              process_node);
    };
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
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
                          mesh::DomainIdRange_c auto&&                                  domain_ids,
                          const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                          const std::array< dofind_t, params.n_equations >&             dof_inds,
                          const SolutionManager::FieldValueGetter< n_fields >&          field_val_getter,
                          std::span< val_t >                                            values,
                          val_t                                                         time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = checkDomainDimension(mesh, domain_ids, params.dimension);
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the domain");

    const auto num_parents     = initValsAndParents(mesh, domain_ids, dof_map, dof_inds, field_val_getter, values);
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
                for (size_t i = 0; auto dof : getNodeDofsAtInds(dof_map, dof_inds, node))
                    if constexpr (n_fields != 0)
                    {
                        const auto np_fp        = static_cast< double >(num_parents[dof]);
                        const auto update_value = ker_res[i++] / np_fp;
                        std::atomic_ref{values[dof]}.fetch_add(update_value, std::memory_order_relaxed);
                    }
                    else
                        std::atomic_ref{values[dof]}.store(ker_res[i++], std::memory_order_relaxed);
            };
            std::ranges::for_each(
                std::views::iota(size_t{}, el_nodes.size()) |
                    std::views::filter([&](size_t node_ind) { return not mesh.isGhostNode(el_nodes[node_ind]); }),
                process_node);
        }
    };
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < typename Kernel,
           KernelParams params,
           el_o_t... orders,
           size_t        max_dofs_per_node,
           std::integral dofind_t,
           size_t        num_maps,
           size_t        n_fields >
void computeValuesAtBoundaryNodes(const ResidualBoundaryKernel< Kernel, params >&               kernel,
                                  const mesh::BoundaryView< orders... >&                        boundary,
                                  const dofs::NodeToLocalDofMap< max_dofs_per_node, num_maps >& dof_map,
                                  const std::array< dofind_t, params.n_equations >&             dof_inds,
                                  const SolutionManager::FieldValueGetter< n_fields >&          field_val_getter,
                                  std::span< val_t >                                            values,
                                  val_t                                                         time = 0.)
{
    L3STER_PROFILE_FUNCTION;

    const bool dim_match = checkBoundaryDimension(boundary, params.dimension);
    util::throwingAssert(dim_match, "The dimension of the kernel does not match the dimension of the boundary");

    const auto num_parents     = initValsAndParents(boundary, dof_map, dof_inds, field_val_getter, values);
    const auto process_element = [&]< mesh::ElementType ET, el_o_t EO >(mesh::BoundaryElementView< ET, EO > el_view) {
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
                for (size_t i = 0; auto dof : getNodeDofsAtInds(dof_map, dof_inds, node))
                {
                    const auto np_fp        = static_cast< double >(num_parents[dof]);
                    const auto update_value = ker_res[i++] / np_fp;
                    std::atomic_ref{values[dof]}.fetch_add(update_value, std::memory_order_relaxed);
                }
            };
            std::ranges::for_each(el_view.getSideNodeInds() | std::views::filter([&](size_t node_ind) {
                                      return not boundary.getParent()->isGhostNode(el_nodes[node_ind]);
                                  }),
                                  process_node);
        }
    };
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr::glob_asm
#endif // L3STER_GLOB_ASM_COMPUTEVALUESATNODES_HPP
