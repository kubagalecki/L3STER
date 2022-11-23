#ifndef L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
#define L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP

#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/basisfun/ReferenceBasisAtNodes.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"

#include <atomic>

namespace lstr
{
namespace detail
{
template < typename Kernel, size_t n_fields, dim_t dim, size_t result_size >
concept ValueAtNodeKernel_c =
    requires(Kernel                                           kernel,
             std::array< val_t, n_fields >                    vals,
             std::array< std::array< val_t, n_fields >, dim > ders,
             SpaceTimePoint                                   p) {
        {
            std::invoke(kernel, vals, ders, p)
            } -> EigenVector_c;
    } and
    (std::invoke_result_t< Kernel,
                           std::array< val_t, n_fields >,
                           std::array< std::array< val_t, n_fields >, dim >,
                           SpaceTimePoint >::RowsAtCompileTime == result_size);

template < typename Kernel, size_t n_fields, dim_t dim, size_t result_size >
concept ValueAtNodeBoundaryKernel_c =
    requires(Kernel                                           kernel,
             std::array< val_t, n_fields >                    vals,
             std::array< std::array< val_t, n_fields >, dim > ders,
             SpaceTimePoint                                   p,
             Eigen::Vector< val_t, dim >                      normal) {
        {
            std::invoke(kernel, vals, ders, p, normal)
            } -> EigenVector_c;
    } and
    (std::invoke_result_t< Kernel,
                           std::array< val_t, n_fields >,
                           std::array< std::array< val_t, n_fields >, dim >,
                           SpaceTimePoint,
                           Eigen::Vector< val_t, dim > >::RowsAtCompileTime == result_size);

template < typename Kernel, size_t n_fields, size_t results_size >
struct PotentiallyValidNodalKernelDeductionHelper
{
    template < ElementTypes T, el_o_t O >
    struct DeductionHelperDomain
    {
        static constexpr bool value =
            ValueAtNodeKernel_c< Kernel, n_fields, Element< T, O >::native_dim, results_size >;
    };
    static constexpr bool domain = assert_any_element< DeductionHelperDomain >;

    template < ElementTypes T, el_o_t O >
    struct DeductionHelperBoundary
    {
        static constexpr bool value =
            ValueAtNodeBoundaryKernel_c< Kernel, n_fields, Element< T, O >::native_dim, results_size >;
    };
    static constexpr bool boundary = assert_any_element< DeductionHelperBoundary >;
};
template < typename Kernel, size_t n_fields, size_t results_size >
concept PotentiallyValidNodalKernel_c =
    PotentiallyValidNodalKernelDeductionHelper< Kernel, n_fields, results_size >::domain;
template < typename Kernel, size_t n_fields, size_t results_size >
concept PotentiallyValidBoundaryNodalKernel_c =
    PotentiallyValidNodalKernelDeductionHelper< Kernel, n_fields, results_size >::boundary;

template < size_t max_dofs_per_node, IndexRange_c auto dof_inds >
auto initValsAndParents(const MeshPartition&                          mesh,
                        detail::DomainIdRange_c auto&&                domain_ids,
                        const NodeToLocalDofMap< max_dofs_per_node >& map,
                        ConstexprValue< dof_inds >                    dofinds_ctwrpr,
                        detail::FieldValGetter_c auto&&               field_val_getter,
                        std::span< val_t >                            values) -> std::vector< std::uint8_t >
{
    if constexpr (deduce_n_fields< decltype(field_val_getter) > != 0)
    {
        std::vector< std::uint8_t > retval(values.size(), 0);
        const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            for (auto node : element.getNodes())
                if (not mesh.isGhostNode(node))
                    for (auto dof : getValuesAtInds(map(node), dofinds_ctwrpr))
                    {
                        std::atomic_ref{retval[dof]}.fetch_add(1, std::memory_order_relaxed);
                        std::atomic_ref{values[dof]}.store(0., std::memory_order_relaxed);
                    }
        };
        mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
        return retval;
    }
    else
        return {};
}

template < size_t max_dofs_per_node, IndexRange_c auto dof_inds >
auto initValsAndParents(const BoundaryView&                           boundary,
                        const NodeToLocalDofMap< max_dofs_per_node >& map,
                        ConstexprValue< dof_inds >                    dofinds_ctwrpr,
                        detail::FieldValGetter_c auto&&               field_val_getter,
                        std::span< val_t >                            values) -> std::vector< std::uint8_t >
{
    std::vector< std::uint8_t > retval(values.size(), 0);
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        for (auto node :
             el_view.getSideNodeInds() | std::views::transform([&](auto ind) { return el_view->getNodes()[ind]; }))
            if (not boundary.getParent()->isGhostNode(node))
                for (auto dof : getValuesAtInds(map(node), dofinds_ctwrpr))
                {
                    std::atomic_ref{retval[dof]}.fetch_add(1, std::memory_order_relaxed);
                    std::atomic_ref{values[dof]}.store(0., std::memory_order_relaxed);
                }
    };
    boundary.visit(process_element, std::execution::par);
    return retval;
}
} // namespace detail

template < size_t max_dofs_per_node, IndexRange_c auto dof_inds >
void computeValuesAtNodes(const MeshPartition&                                  mesh,
                          detail::DomainIdRange_c auto&&                        domain_ids,
                          const NodeToLocalDofMap< max_dofs_per_node >&         map,
                          ConstexprValue< dof_inds >                            dofinds_ctwrpr,
                          std::span< const val_t, std::ranges::size(dof_inds) > values_in,
                          std::span< val_t >                                    values_out)
    requires(std::ranges::all_of(dof_inds, [](size_t dof) { return dof < max_dofs_per_node; }))
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto& el_nodes     = element.getNodes();
        const auto  process_node = [&](size_t node_ind) {
            for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                std::atomic_ref{values_out[dof]}.store(values_in[dof_ind++], std::memory_order_relaxed);
        };
        for (size_t node_ind = 0; node_ind < el_nodes.size(); ++node_ind)
            if (not mesh.isGhostNode(el_nodes[node_ind]))
                process_node(node_ind);
    };
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < size_t max_dofs_per_node, IndexRange_c auto dof_inds >
void computeValuesAtNodes(auto&&                                        kernel,
                          const MeshPartition&                          mesh,
                          detail::DomainIdRange_c auto&&                domain_ids,
                          const NodeToLocalDofMap< max_dofs_per_node >& map,
                          ConstexprValue< dof_inds >                    dofinds_ctwrpr,
                          detail::FieldValGetter_c auto&&               field_val_getter,
                          std::span< val_t >                            values,
                          val_t                                         time = 0.)
    requires detail::PotentiallyValidNodalKernel_c< decltype(kernel),
                                                    detail::deduce_n_fields< decltype(field_val_getter) >,
                                                    std::ranges::size(dof_inds) > and
             (std::ranges::all_of(dof_inds, [](size_t dof) { return dof < max_dofs_per_node; }))
{
    const auto num_parents =
        detail::initValsAndParents(mesh, domain_ids, map, dofinds_ctwrpr, field_val_getter, values);
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        if constexpr (detail::ValueAtNodeKernel_c< decltype(kernel),
                                                   detail::deduce_n_fields< decltype(field_val_getter) >,
                                                   Element< ET, EO >::native_dim,
                                                   std::ranges::size(dof_inds) >)
        {
            const auto& el_nodes       = element.getNodes();
            const auto& basis_at_nodes = getBasisAtNodes< ET, EO >();
            const auto& node_locations = getNodeLocations< ET, EO >();

            const auto node_vals            = field_val_getter(el_nodes);
            const auto jacobi_mat_generator = getNatJacobiMatGenerator(element);

            const auto process_node = [&](size_t node_ind) {
                const auto ref_coords      = node_locations[node_ind];
                const auto jacobi_mat      = jacobi_mat_generator(ref_coords);
                const auto phys_coords     = mapToPhysicalSpace(element, ref_coords);
                const auto phys_basis_ders = computePhysBasisDers(jacobi_mat, basis_at_nodes.derivatives[node_ind]);
                const auto field_vals      = detail::computeFieldVals(basis_at_nodes.values[node_ind], node_vals);
                const auto field_ders      = detail::computeFieldDers(phys_basis_ders, node_vals);
                const auto ker_res = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time});
                for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                    if constexpr (detail::deduce_n_fields< decltype(field_val_getter) > != 0)
                        std::atomic_ref{values[dof]}.fetch_add(
                            ker_res[dof_ind++] / static_cast< double >(num_parents[dof]), std::memory_order_relaxed);
                    else
                        std::atomic_ref{values[dof]}.store(ker_res[dof_ind++], std::memory_order_relaxed);
            };
            for (size_t node_ind = 0; node_ind < el_nodes.size(); ++node_ind)
                if (not mesh.isGhostNode(el_nodes[node_ind]))
                    process_node(node_ind);
        }
        else
        {
            std::cerr << "Attempting to compute nodal values for an element for which the passed kernel is invalid. "
                         "Please check the kernel was defined correctly, and that you are setting nodal values in the "
                         "correct domain (e.g. that you're not trying to evaluate a 2D kernel in a 3D domain). This "
                         "process will now terminate.\n";
            std::terminate();
        }
    };
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < size_t max_dofs_per_node, IndexRange_c auto dof_inds >
void computeValuesAtBoundaryNodes(auto&&                                        kernel,
                                  const BoundaryView&                           boundary,
                                  const NodeToLocalDofMap< max_dofs_per_node >& map,
                                  ConstexprValue< dof_inds >                    dofinds_ctwrpr,
                                  detail::FieldValGetter_c auto&&               field_val_getter,
                                  std::span< val_t >                            values,
                                  val_t                                         time = 0.)
    requires detail::PotentiallyValidBoundaryNodalKernel_c< decltype(kernel),
                                                            detail::deduce_n_fields< decltype(field_val_getter) >,
                                                            std::ranges::size(dof_inds) > and
             (std::ranges::all_of(dof_inds, [](size_t dof) { return dof < max_dofs_per_node; }))
{
    const auto num_parents     = detail::initValsAndParents(boundary, map, dofinds_ctwrpr, field_val_getter, values);
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(BoundaryElementView< ET, EO > el_view) {
        if constexpr (detail::ValueAtNodeBoundaryKernel_c< decltype(kernel),
                                                           detail::deduce_n_fields< decltype(field_val_getter) >,
                                                           Element< ET, EO >::native_dim,
                                                           std::ranges::size(dof_inds) >)
        {
            const auto& el_nodes       = el_view->getNodes();
            const auto& basis_at_nodes = getBasisAtNodes< ET, EO >();
            const auto& node_locations = getNodeLocations< ET, EO >();

            const auto node_vals            = field_val_getter(el_nodes);
            const auto jacobi_mat_generator = getNatJacobiMatGenerator(*el_view);

            const auto process_node = [&](size_t node_ind) {
                const auto ref_coords      = node_locations[node_ind];
                const auto jacobi_mat      = jacobi_mat_generator(ref_coords);
                const auto phys_coords     = mapToPhysicalSpace(*el_view, ref_coords);
                const auto normal          = computeBoundaryNormal(el_view, jacobi_mat);
                const auto phys_basis_ders = computePhysBasisDers(jacobi_mat, basis_at_nodes.derivatives[node_ind]);
                const auto field_vals      = detail::computeFieldVals(basis_at_nodes.values[node_ind], node_vals);
                const auto field_ders      = detail::computeFieldDers(phys_basis_ders, node_vals);
                const auto ker_res =
                    std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time}, normal);
                for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                    std::atomic_ref{values[dof]}.fetch_add(ker_res[dof_ind++] / static_cast< double >(num_parents[dof]),
                                                           std::memory_order_relaxed);
            };
            for (auto node_ind : el_view.getSideNodeInds())
                if (not boundary.getParent()->isGhostNode(el_nodes[node_ind]))
                    process_node(node_ind);
        }
        else
        {
            std::cerr << "Attempting to compute boundary nodal values for an element for which the passed kernel is "
                         "invalid. Please check the kernel was defined correctly, and that you are setting nodal "
                         "values in the correct domain (e.g. that you're not trying to evaluate a 2D kernel in a 3D "
                         "domain). This process will now terminate.\n";
            std::terminate();
        }
    };
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
