#ifndef L3STER_ASSEMBLY_NODETODOFMAP_HPP
#define L3STER_ASSEMBLY_NODETODOFMAP_HPP

#include "l3ster/assembly/DofIntervals.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include "Tpetra_CrsGraph.hpp"

namespace lstr
{
template < size_t dofs_per_node >
class NodeToGlobalDofMap
{
    using payload_t = std::array< global_dof_t, dofs_per_node >;
    struct ContiguousCaseInfo
    {
        static_assert(dofs_per_node <= (1u << (sizeof(std::uint8_t) * CHAR_BIT)));
        n_id_t                                    base_node;
        global_dof_t                              base_dof;
        std::array< std::uint8_t, dofs_per_node > dof_inds;
        std::uint8_t                              n_dofs;
    };
    using map_t = robin_hood::unordered_flat_map< n_id_t, payload_t >;

public:
    using dof_t                               = global_dof_t;
    static constexpr global_dof_t invalid_dof = -1;

    NodeToGlobalDofMap() = default;
    template < CondensationPolicy CP >
    NodeToGlobalDofMap(const MeshPartition&                                   mesh,
                       const detail::node_interval_vector_t< dofs_per_node >& dof_intervals,
                       const detail::NodeCondensationMap< CP >&               cond_map);

    [[nodiscard]] inline payload_t operator()(n_id_t node) const noexcept;
    [[nodiscard]] bool             isContiguous() const noexcept
    {
        return std::get_if< ContiguousCaseInfo >(std::addressof(m_data));
    }

private:
    template < CondensationPolicy CP >
    bool tryInitAsContiguous(const MeshPartition&                                   msh,
                             const detail::node_interval_vector_t< dofs_per_node >& dof_ints,
                             const detail::NodeCondensationMap< CP >&               cond_map);
    template < CondensationPolicy CP >
    void initNonContiguous(const MeshPartition&                                   mesh,
                           const detail::node_interval_vector_t< dofs_per_node >& dof_intervals,
                           const detail::NodeCondensationMap< CP >&               cond_map);

    std::variant< map_t, ContiguousCaseInfo > m_data;
};

template < size_t dofs_per_node, size_t num_maps >
class NodeToLocalDofMap
{
    using payload_t = std::array< std::array< local_dof_t, dofs_per_node >, num_maps >;
    using map_t     = robin_hood::unordered_flat_map< n_id_t, payload_t >;

public:
    static constexpr local_dof_t invalid_dof = -1;
    using dof_t                              = local_dof_t;

    NodeToLocalDofMap() = default;
    NodeToLocalDofMap(const MeshPartition&                       nodes,
                      const NodeToGlobalDofMap< dofs_per_node >& global_map,
                      const std::same_as< tpetra_map_t > auto&... local_global_maps)
        requires(sizeof...(local_global_maps) == num_maps);
    [[nodiscard]] const payload_t& operator()(n_id_t node) const noexcept { return m_map.find(node)->second; }

private:
    map_t m_map;
};

template < size_t dofs_per_node >
NodeToLocalDofMap(const MeshPartition&                       nodes,
                  const NodeToGlobalDofMap< dofs_per_node >& global_map,
                  const std::same_as< tpetra_map_t > auto&... local_global_maps)
    -> NodeToLocalDofMap< dofs_per_node, sizeof...(local_global_maps) >;

namespace detail
{
template < typename T >
inline constexpr bool is_node_map = false;
template < size_t dpn >
inline constexpr bool is_node_map< NodeToGlobalDofMap< dpn > > = true;
template < size_t dpn, size_t nm >
inline constexpr bool is_node_map< NodeToLocalDofMap< dpn, nm > > = true;
} // namespace detail

template < typename T >
concept NodeToDofMap_c = detail::is_node_map< T >;

template < size_t dofs_per_node >
template < CondensationPolicy CP >
NodeToGlobalDofMap< dofs_per_node >::NodeToGlobalDofMap(
    const MeshPartition&                                   mesh,
    const detail::node_interval_vector_t< dofs_per_node >& dof_intervals,
    const detail::NodeCondensationMap< CP >&               cond_map)
{
    L3STER_PROFILE_FUNCTION;
    if (not tryInitAsContiguous(mesh, dof_intervals, cond_map))
        initNonContiguous(mesh, dof_intervals, cond_map);
}

template < size_t dofs_per_node >
typename NodeToGlobalDofMap< dofs_per_node >::payload_t
NodeToGlobalDofMap< dofs_per_node >::operator()(n_id_t node) const noexcept
{
    const auto map_ptr = std::get_if< map_t >(std::addressof(m_data));
    if (map_ptr)
        return map_ptr->find(node)->second;
    else
    {
        const auto& contig_info = *std::get_if< ContiguousCaseInfo >(std::addressof(m_data));
        const auto  dof_inds    = std::span{contig_info.dof_inds.begin(), contig_info.n_dofs};
        auto        node_dof    = contig_info.base_dof + (node - contig_info.base_node) * contig_info.n_dofs;
        payload_t   retval;
        retval.fill(invalid_dof);
        for (auto i : dof_inds)
            retval[i] = node_dof++;
        return retval;
    }
}

template < size_t dofs_per_node >
template < CondensationPolicy CP >
bool NodeToGlobalDofMap< dofs_per_node >::tryInitAsContiguous(
    const MeshPartition&                                   mesh,
    const detail::node_interval_vector_t< dofs_per_node >& dof_intervals,
    const detail::NodeCondensationMap< CP >&               cond_map)
{
    if constexpr (CP == CondensationPolicy::ElementBoundary)
        return false;
    else if constexpr (CP == CondensationPolicy::None)
    {
        const auto min_node_in_partition = std::min(
            mesh.getOwnedNodes().empty() ? std::numeric_limits< n_id_t >::max() : mesh.getOwnedNodes().front(),
            mesh.getGhostNodes().empty() ? std::numeric_limits< n_id_t >::max() : mesh.getGhostNodes().front());
        const auto max_node_in_partition =
            std::max(mesh.getOwnedNodes().empty() ? std::numeric_limits< n_id_t >::min() : mesh.getOwnedNodes().back(),
                     mesh.getGhostNodes().empty() ? std::numeric_limits< n_id_t >::min() : mesh.getGhostNodes().back());
        global_dof_t base_dof{};
        for (const auto& interval : dof_intervals)
        {
            const auto& [delim, coverage]     = interval;
            const auto& [int_first, int_last] = delim;
            if (min_node_in_partition >= int_first and max_node_in_partition <= int_last)
            {
                std::array< std::uint8_t, dofs_per_node > dof_inds{};
                auto                                      write_it = begin(dof_inds);
                for (size_t i = 0; i < dofs_per_node; ++i)
                    if (coverage.test(i))
                        *write_it++ = static_cast< std::uint8_t >(i);
                m_data =
                    ContiguousCaseInfo{int_first, base_dof, dof_inds, static_cast< std::uint8_t >(coverage.count())};
                return true;
            }
            base_dof += (int_last - int_first + 1) * coverage.count();
        }
        return false;
    }
}

template < size_t dofs_per_node >
template < CondensationPolicy CP >
void NodeToGlobalDofMap< dofs_per_node >::initNonContiguous(
    const MeshPartition&                                   mesh,
    const detail::node_interval_vector_t< dofs_per_node >& dof_intervals,
    const detail::NodeCondensationMap< CP >&               cond_map)
{
    auto&      map                 = m_data.template emplace< map_t >();
    const auto dof_interval_starts = detail::computeIntervalStarts(dof_intervals);
    const auto add_entries         = [&](std::span< const n_id_t > nodes) {
        const auto compute_node_dofs = [&](n_id_t node_id, ptrdiff_t interval_ind) {
            const auto dof_int_start = dof_interval_starts[interval_ind];
            const auto& [delim, cov] = dof_intervals[interval_ind];
            const auto& [lo, hi]     = delim;

            std::array< global_dof_t, dofs_per_node > retval;
            retval.fill(invalid_dof);
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
    add_entries(mesh.getOwnedNodes());
    add_entries(mesh.getGhostNodes());
}

template < size_t dofs_per_node, size_t num_maps >
NodeToLocalDofMap< dofs_per_node, num_maps >::NodeToLocalDofMap(
    const MeshPartition&                       mesh,
    const NodeToGlobalDofMap< dofs_per_node >& global_map,
    const std::same_as< tpetra_map_t > auto&... local_global_maps)
    requires(sizeof...(local_global_maps) == num_maps)
    : m_map(mesh.getAllNodes().size())
{
    L3STER_PROFILE_FUNCTION;
    const auto get_node_dofs = [&](n_id_t node, const tpetra_map_t& map) {
        std::array< local_dof_t, dofs_per_node > retval;
        std::ranges::transform(global_map(node), begin(retval), [&](global_dof_t dof) {
            return dof == NodeToGlobalDofMap< dofs_per_node >::invalid_dof ? invalid_dof : map.getLocalElement(dof);
        });
        return retval;
    };
    for (n_id_t node : mesh.getAllNodes())
        m_map[node] = payload_t{get_node_dofs(node, local_global_maps)...};
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_NODETODOFMAP_HPP
