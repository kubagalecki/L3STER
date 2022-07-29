#ifndef L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP
#define L3STER_ASSEMBLY_COMPUTEVALUESATNODES_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/mesh/NodePhysicalLocation.hpp"

#include <atomic>

namespace lstr
{
template < array_of< size_t > auto dof_inds, typename F, std::ranges::sized_range R, size_t n_fields >
auto computeValuesAtNodes(F&&                             f,
                          const MeshPartition&            mesh,
                          R&&                             dom_ids,
                          const NodeToDofMap< n_fields >& map,
                          Tpetra::Vector<>&               values,
                          val_t                           time = 0.)
    requires std::convertible_to< std::ranges::range_value_t< R >,
                                  d_id_t > and
             (std::ranges::all_of(dof_inds, [](size_t dof) { return dof < n_fields; })) and
                 requires(const SpaceTimePoint p)
{
    {
        f(p)
        } -> EigenVector_c;
}
and(std::invoke_result_t< F, SpaceTimePoint >::RowsAtCompileTime == dof_inds.size())
{
    const auto is_owned_node = [&](n_id_t node) {
        return not std::ranges::binary_search(mesh.getGhostNodes(), node);
    };
    const auto  local_vals_view         = values.getDataNonConst();
    const auto& dof_global_to_local_map = *values.getMap();
    const auto  process_element         = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        const auto& el_nodes     = element.getNodes();
        const auto  process_node = [&](size_t node_ind) {
            const auto vals_at_node = f(SpaceTimePoint{.space = nodePhysicalLocation(element, node_ind), .time = time});
            for (size_t dof_ind = 0; auto dof : getValuesAtInds< dof_inds >(map(el_nodes[node_ind])))
            {
                const auto val           = vals_at_node[dof_ind++];
                const auto local_dof_ind = dof_global_to_local_map.getLocalElement(dof);
                std::atomic_ref{local_vals_view[local_dof_ind]}.store(val, std::memory_order_relaxed);
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
