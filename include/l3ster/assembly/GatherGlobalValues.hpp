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
    constexpr auto                                    num_fields = std::ranges::size(field_inds);
    EigenRowMajorMatrix< val_t, n_nodes, num_fields > retval;
    for (size_t node_ind = 0; auto node : nodes)
    {
        const auto node_vals = solution_manager.getNodeValues(node, std::span{field_inds});
        std::ranges::copy(node_vals, std::next(retval.data(), num_fields * node_ind++));
    }
    return retval;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_GATHERGLOBALVALUES_HPP
