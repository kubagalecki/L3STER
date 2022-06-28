#ifndef L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
#define L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/util/IncludeEigen.hpp"

namespace lstr
{
template < array_of< size_t > auto dof_inds, ElementTypes T, el_o_t O, size_t n_fields >
auto gatherGlobalValues(const Element< T, O >&          element,
                        const NodeToDofMap< n_fields >& map,
                        const Tpetra::Vector<>&         global_values)
{
    static_assert(std::ranges::all_of(dof_inds, [](auto dof) { return dof < n_fields; }));

    const auto vals_view = global_values.getData();
    const auto el_dofs   = detail::getUnsortedElementDofs< dof_inds >(element, map);
    Eigen::Matrix< val_t,
                   Element< T, O >::n_nodes,
                   dof_inds.size(),
                   dof_inds.size() == 1 ? Eigen::ColMajor : Eigen::RowMajor >
        retval;
    std::ranges::transform(el_dofs, retval.data(), [&](global_dof_t dof) {
        return vals_view[global_values.getMap()->getLocalElement(dof)];
    });
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
