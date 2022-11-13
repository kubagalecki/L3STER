#ifndef L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
#define L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP

#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/basisfun/ReferenceBasisAtNodes.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"

#include <atomic>

namespace lstr
{
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
void computeValuesAtNodes(auto&&                               f,
                          const MeshPartition&                 mesh,
                          detail::DomainIdRange_c auto&&       domain_ids,
                          const NodeToLocalDofMap< n_fields >& map,
                          ConstexprValue< dof_inds >           dofinds_ctwrpr,
                          detail::FieldValGetter_c auto&&      field_val_getter,
                          std::span< val_t >                   values,
                          val_t                                time = 0.)
    requires(std::ranges::all_of(dof_inds, [](size_t dof) { return dof < n_fields; })) and
            requires(const SpaceTimePoint p) {
                {
                    f(p)
                    } -> EigenVector_c;
            } and (std::invoke_result_t< decltype(f), SpaceTimePoint >::RowsAtCompileTime == dof_inds.size())
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto&                 el_nodes  = element.getNodes();
        [[maybe_unused]] const auto node_vals = field_val_getter(el_nodes);

        const auto process_node = [&](size_t node_ind) {
            const auto physical_point = SpaceTimePoint{.space = nodePhysicalLocation(element, node_ind), .time = time};
            const auto vals_at_node   = f(physical_point);
            for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                std::atomic_ref{values[dof]}.store(vals_at_node[dof_ind++], std::memory_order_relaxed);
        };
        for (size_t node_ind = 0; node_ind < el_nodes.size(); ++node_ind)
            if (not mesh.isGhostNode(el_nodes[node_ind]))
                process_node(node_ind);
    };
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}

template < size_t n_fields, IndexRange_c auto dof_inds >
void computeBoundaryValuesAtNodes(const BoundaryView&                                   boundary,
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
            if (not boundary.getParent()->isGhostNode(el_nodes[node_ind]))
                process_node(node_ind);
    };
    boundary.visit(process_element, std::execution::par);
}

template < size_t n_fields, IndexRange_c auto dof_inds >
auto computeBoundaryValuesAtNodes(auto&&                               f,
                                  const BoundaryView&                  boundary,
                                  const NodeToLocalDofMap< n_fields >& map,
                                  ConstexprValue< dof_inds >           dofinds_ctwrpr,
                                  std::span< val_t >                   values,
                                  val_t                                time = 0.)
    requires(std::ranges::all_of(dof_inds, [](size_t dof) { return dof < n_fields; }))
{
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const BoundaryElementView< ET, EO >& el_view) {
        const auto& el_nodes     = el_view->getNodes();
        const auto  process_node = [&](size_t node_ind) {
            const auto vals_at_node = f(SpaceTimePoint{.space = nodePhysicalLocation(el_view, node_ind), .time = time});
            for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                std::atomic_ref{values[dof]}.store(vals_at_node[dof_ind++], std::memory_order_relaxed);
        };
        for (size_t node_ind = 0; node_ind < el_nodes.size(); ++node_ind)
            if (not boundary.getParent()->isGhostNode(el_nodes[node_ind]))
                process_node(node_ind);
    };
    boundary.visit(process_element, std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
