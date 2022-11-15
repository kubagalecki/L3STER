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
concept ValueAtNodeKernel =
    requires(Kernel                                                 kernel,
             const std::array< val_t, n_fields >                    vals,
             const std::array< std::array< val_t, n_fields >, dim > ders,
             const SpaceTimePoint                                   p) {
        {
            std::invoke(kernel, vals, ders, p)
            } -> EigenVector_c;
    } and
    (std::invoke_result_t< Kernel,
                           const std::array< val_t, n_fields >,
                           const std::array< const std::array< val_t, n_fields >, dim >,
                           const SpaceTimePoint >::RowsAtCompileTime == result_size);
template < typename Kernel, size_t n_fields, dim_t dim, size_t result_size >

concept ValueAtNodeBoundaryKernel =
    requires(Kernel                                                 kernel,
             const std::array< val_t, n_fields >                    vals,
             const std::array< std::array< val_t, n_fields >, dim > ders,
             const SpaceTimePoint                                   p,
             const Eigen::Vector< val_t, dim >                      normal) {
        {
            std::invoke(kernel, vals, ders, p, normal)
            } -> EigenVector_c;
    } and
    (std::invoke_result_t< Kernel,
                           const std::array< val_t, n_fields >,
                           const std::array< const std::array< val_t, n_fields >, dim >,
                           const SpaceTimePoint >::RowsAtCompileTime == result_size);
} // namespace detail

template < size_t n_fields, IndexRange_c auto dof_inds >
void computeValuesAtNodes(const MeshPartition&                                  mesh,
                          detail::DomainIdRange_c auto&&                        domain_ids,
                          const NodeToLocalDofMap< n_fields >&                  map,
                          ConstexprValue< dof_inds >                            dofinds_ctwrpr,
                          std::span< const val_t, std::ranges::size(dof_inds) > values_in,
                          std::span< val_t >                                    values_out)
    requires(std::ranges::all_of(dof_inds, [](size_t dof) { return dof < n_fields; }))
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

template < size_t n_fields, IndexRange_c auto dof_inds >
void computeValuesAtNodes(auto&&                               kernel,
                          const MeshPartition&                 mesh,
                          detail::DomainIdRange_c auto&&       domain_ids,
                          const NodeToLocalDofMap< n_fields >& map,
                          ConstexprValue< dof_inds >           dofinds_ctwrpr,
                          detail::FieldValGetter_c auto&&      field_val_getter,
                          std::span< val_t >                   values,
                          val_t                                time = 0.)
    requires(std::ranges::all_of(dof_inds, [](size_t dof) { return dof < n_fields; }))
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        if constexpr (detail::ValueAtNodeKernel< decltype(kernel),
                                                 detail::deduce_n_fields< decltype(field_val_getter) >,
                                                 Element< ET, EO >::native_dim,
                                                 std::ranges::size(dof_inds) >)
        {
            const auto& el_nodes       = element.getNodes();
            const auto& basis_at_nodes = getBasisAtNodes< ET, EO >();
            const auto& node_locations = getNodeLocations< ET, EO >();
            const auto& basis_vals     = basis_at_nodes.values;

            const auto node_vals           = field_val_getter(el_nodes);
            const auto jac_at_nodes        = computeJacobiansAtPoints(element, node_locations);
            const auto basis_ders          = computePhysBasisDersAtPoints(basis_at_nodes.derivatives, jac_at_nodes);
            const auto field_vals_and_ders = detail::computeFieldValsAndDers(basis_vals, basis_ders, node_vals);

            const auto process_node = [&](size_t node_ind) {
                const auto [vals, ders] = detail::extractFieldValsAndDersAtQpoint(field_vals_and_ders, node_ind);
                const auto physical_point =
                    SpaceTimePoint{.space = nodePhysicalLocation(element, node_ind), .time = time};
                const auto vals_at_node = std::invoke(kernel, vals, ders, physical_point);
                for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                    std::atomic_ref{values[dof]}.store(vals_at_node[dof_ind++], std::memory_order_relaxed);
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
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
