#ifndef L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
#define L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/util/IncludeEigen.hpp"

namespace lstr
{
template < array_of< size_t > auto dof_inds, size_t n_nodes, size_t n_fields >
auto gatherGlobalValues(const std::array< n_id_t, n_nodes >&            el_nodes,
                        const NodeToDofMap< n_fields >&                 map,
                        const Teuchos::ArrayRCP< const val_t >&         global_values,
                        const Tpetra::Map< local_dof_t, global_dof_t >& dof_local_global_map)
{
    static_assert(std::ranges::all_of(dof_inds, [](auto dof) { return dof < n_fields; }));

    const auto el_dofs = detail::getDofsFromNodes< dof_inds >(el_nodes, map);
    Eigen::Matrix< val_t, n_nodes, dof_inds.size(), dof_inds.size() == 1 ? Eigen::ColMajor : Eigen::RowMajor > retval;
    std::ranges::transform(el_dofs, retval.data(), [&](global_dof_t dof) {
        return global_values[dof_local_global_map.getLocalElement(dof)];
    });
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
