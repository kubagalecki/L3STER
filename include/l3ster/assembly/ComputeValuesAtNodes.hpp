#ifndef L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
#define L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"

#include <atomic>

namespace lstr
{
template < typename F, detail::DomainIdRange_c R, size_t n_fields, IndexRange_c auto dof_inds >
auto computeValuesAtNodes(F&&                                                 f,
                          const MeshPartition&                                mesh,
                          R&&                                                 dom_ids,
                          const NodeToLocalDofMap< n_fields >&                map,
                          ConstexprValue< dof_inds >                          dofinds_ctwrpr,
                          Tpetra::Vector< val_t, local_dof_t, global_dof_t >& values,
                          val_t                                               time = 0.)
    requires(std::ranges::all_of(dof_inds, [](size_t dof) { return dof < n_fields; })) and
            requires(const SpaceTimePoint p) {
                {
                    f(p)
                    } -> EigenVector_c;
            } and (std::invoke_result_t< F, SpaceTimePoint >::RowsAtCompileTime == dof_inds.size())
{
    const auto is_owned_node = [&](n_id_t node) {
        return not std::ranges::binary_search(mesh.getGhostNodes(), node);
    };
    const auto local_vals_alloc = values.getDataNonConst();
    std::span  local_vals_view{local_vals_alloc};
    const auto process_element = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto& el_nodes     = element.getNodes();
        const auto  process_node = [&](size_t node_ind) {
            const auto vals_at_node = f(SpaceTimePoint{.space = nodePhysicalLocation(element, node_ind), .time = time});
            for (size_t dof_ind = 0; auto dof : getValuesAtInds(map(el_nodes[node_ind]), dofinds_ctwrpr))
            {
                const auto val = vals_at_node[dof_ind++];
                std::atomic_ref{local_vals_view[dof]}.store(val, std::memory_order_relaxed);
            }
        };
        for (size_t node_ind = 0; node_ind < el_nodes.size(); ++node_ind)
            if (is_owned_node(el_nodes[node_ind]))
                process_node(node_ind);
    };
    mesh.visit(process_element, std::forward< R >(dom_ids), std::execution::par);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
