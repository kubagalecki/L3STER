#ifndef L3STER_ASSEMBLY_NODETODOFMAP_HPP
#define L3STER_ASSEMBLY_NODETODOFMAP_HPP

#include "l3ster/assembly/DofIntervals.hpp"

#include "Tpetra_CrsGraph.hpp"

namespace lstr
{
template < size_t NF > // number of fields
class NodeToDofMap
{
public:
    NodeToDofMap() = default;
    NodeToDofMap(const MeshPartition& mesh, const detail::node_interval_vector_t< NF >& dof_intervals);

    [[nodiscard]] const auto& operator()(n_id_t node) const noexcept { return map.find(node)->second; }

private:
    std::unordered_map< n_id_t, std::array< global_dof_t, NF > > map;
};

template < size_t NF >
NodeToDofMap< NF >::NodeToDofMap(const MeshPartition& mesh, const detail::node_interval_vector_t< NF >& dof_intervals)
    : map(mesh.getNodes().size() + mesh.getGhostNodes().size())
{
    const auto dof_interval_starts = detail::computeIntervalStarts(dof_intervals);
    const auto add_entries         = [&](const std::vector< n_id_t >& nodes) {
        const auto compute_node_dofs = [&](n_id_t node_id, ptrdiff_t interval_ind) {
            const auto dof_int_start = dof_interval_starts[interval_ind];
            const auto& [delim, cov] = dof_intervals[interval_ind];
            const auto& [lo, hi]     = delim;

            std::array< global_dof_t, NF > retval;
            retval.fill(std::numeric_limits< global_dof_t >::max());
            global_dof_t node_dof = dof_int_start + (node_id - lo) * cov.count();
            for (ptrdiff_t i = 0; auto& dof : retval)
                if (cov.test(i++))
                    dof = node_dof++;
            return retval;
        };

        for (auto search_it = begin(dof_intervals); auto n : nodes)
        {
            search_it               = detail::findNodeInterval(search_it, end(dof_intervals), n);
            const auto interval_ind = std::distance(begin(dof_intervals), search_it);
            const auto node_dofs    = compute_node_dofs(n, interval_ind);
            map.emplace(n, node_dofs);
        }
    };
    add_entries(mesh.getNodes());
    add_entries(mesh.getGhostNodes());
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_NODETODOFMAP_HPP
