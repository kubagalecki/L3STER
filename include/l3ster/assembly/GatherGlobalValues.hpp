#ifndef L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
#define L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/util/IncludeEigen.hpp"

namespace lstr
{
template < IndexRange_c auto dof_inds, size_t n_nodes, size_t n_fields >
auto gatherGlobalValues(const std::array< n_id_t, n_nodes >& el_nodes,
                        const NodeToLocalDofMap< n_fields >& map,
                        std::span< const val_t >             global_values,
                        ConstexprValue< dof_inds >           dofinds_ctwrpr = {})
    requires(std::ranges::all_of(dof_inds, [](auto dof) { return dof < n_fields; }))
{
    const auto el_dofs = detail::getDofsFromNodes< dof_inds >(el_nodes, map);
    EigenRowMajorMatrix< val_t, n_nodes, dof_inds.size() > retval;
    std::ranges::transform(el_dofs, retval.data(), [&](global_dof_t dof) { return global_values[dof]; });
    return retval;
}

template < size_t n_nodes, size_t n_fields >
auto gatherGlobalValues(const std::array< n_id_t, n_nodes >&                  el_nodes,
                        Mapping_c< n_id_t, local_dof_t > auto&&               map,
                        std::span< const std::span< const val_t >, n_fields > global_values)
{
    EigenRowMajorMatrix< val_t, n_nodes, n_fields > retval;
    for (size_t i = 0; auto node : el_nodes)
    {
        const auto val_ind = map(node);
        for (size_t field = 0; field < n_fields; ++field)
            retval(i, field) = global_values[field][val_ind];
        ++i;
    }
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
