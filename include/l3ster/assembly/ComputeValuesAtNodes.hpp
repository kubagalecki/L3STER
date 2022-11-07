#ifndef L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
#define L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"

#include <atomic>

namespace lstr
{
template < size_t n_fields, IndexRange_c auto dof_inds >
auto computeValuesAtNodes(auto&&                               f,
                          const MeshPartition&                 mesh,
                          detail::DomainIdRange_c auto&&       domain_ids,
                          const NodeToLocalDofMap< n_fields >& map,
                          ConstexprValue< dof_inds >           dofinds_ctwrpr,
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
        const auto& el_nodes     = element.getNodes();
        const auto  process_node = [&](size_t node_ind) {
            const auto vals_at_node = f(SpaceTimePoint{.space = nodePhysicalLocation(element, node_ind), .time = time});
            for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
                std::atomic_ref{values[dof]}.store(vals_at_node[dof_ind++], std::memory_order_relaxed);
        };
        for (size_t node_ind = 0; node_ind < el_nodes.size(); ++node_ind)
            if (not mesh.isGhostNode(el_nodes[node_ind]))
                process_node(node_ind);
    };
    mesh.visit(process_element, std::forward< decltype(domain_ids) >(domain_ids), std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
