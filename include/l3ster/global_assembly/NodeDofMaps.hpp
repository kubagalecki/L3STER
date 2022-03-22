#ifndef L3STER_ASSEMBLY_NODEDOFMAPS_HPP
#define L3STER_ASSEMBLY_NODEDOFMAPS_HPP

#include "l3ster/global_assembly/DofIntervals.hpp"
#include "l3ster/global_assembly/NodeLocalGlobalConverter.hpp"

namespace lstr
{
template < size_t NF > // number of fields
class GlobalNodeToDofMap
{
public:
    GlobalNodeToDofMap() = default;
    GlobalNodeToDofMap(const MeshPartition& mesh, const detail::node_interval_vector_t< NF >& dof_intervals);

    [[nodiscard]] auto operator()(n_id_t node) const { return map.find(node)->second; }

private:
    std::unordered_map< n_id_t, std::array< global_dof_t, NF > > map;
};

template < size_t NF >
class LocalNodeToDofMap
{
public:
    LocalNodeToDofMap() = default;
    LocalNodeToDofMap(const MeshPartition& mesh, const detail::node_interval_vector_t< NF >& dof_intervals);

    [[nodiscard]] auto operator()(n_id_t node) const { return map[node]; }

private:
    std::vector< std::array< local_dof_t, NF > > map;
};

template < size_t NF >
GlobalNodeToDofMap< NF >::GlobalNodeToDofMap(const MeshPartition&                        mesh,
                                             const detail::node_interval_vector_t< NF >& dof_intervals)
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

template < size_t NF >
LocalNodeToDofMap< NF >::LocalNodeToDofMap(const MeshPartition&                        mesh,
                                           const detail::node_interval_vector_t< NF >& dof_intervals)
    : map(mesh.getNodes().size() + mesh.getGhostNodes().size())
{
    auto add_entries = [&, entry = n_id_t{0}, dof = global_dof_t{0}](const std::vector< n_id_t >& nodes) mutable {
        for (auto search_it = begin(dof_intervals); auto n : nodes)
        {
            search_it = detail::findNodeInterval(search_it, end(dof_intervals), n);
            std::array< local_dof_t, NF > node_dofs;
            node_dofs.fill(std::numeric_limits< local_dof_t >::max());
            const auto& cov = search_it->second;
            for (size_t i = 0; i < NF; ++i)
                if (cov.test(i))
                    node_dofs[i] = dof++;
            map[entry++] = node_dofs;
        }
    };
    add_entries(mesh.getNodes());
    add_entries(mesh.getGhostNodes());
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_NODEDOFMAPS_HPP
