#ifndef L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
#define L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP

#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/util/IncludeEigen.hpp"

namespace lstr
{
template < IndexRange_c auto field_inds, size_t n_nodes >
auto gatherGlobalValues(const std::array< n_id_t, n_nodes >& nodes,
                        const SolutionManager&               solution_manager,
                        ConstexprValue< field_inds > = {})
{
    EigenRowMajorMatrix< val_t, n_nodes, std::ranges::size(field_inds) > retval;
    for (size_t node_ind = 0; auto node : nodes)
    {
        const auto local_node_index = solution_manager.getNodeMap().getLocalElement(node);
        for (size_t i = 0; auto field_ind : field_inds)
            retval(node_ind, i++) = solution_manager.getNodalValues(field_ind)[local_node_index];
        ++node_ind;
    }
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
