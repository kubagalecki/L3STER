#ifndef L3STER_SOLUTIONMANAGER_HPP
#define L3STER_SOLUTIONMANAGER_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#include <memory>

namespace lstr
{
class SolutionManager
{
public:
    using node_map_t = robin_hood::unordered_flat_map< n_id_t, ptrdiff_t >;

    inline SolutionManager(const MeshPartition& mesh, size_t n_fields, val_t initial = 0.);

    [[nodiscard]] auto        nFields() const -> size_t { return m_n_fields; }
    [[nodiscard]] auto        nNodes() const -> size_t { return m_n_nodes; }
    [[nodiscard]] inline auto getFieldView(size_t field_ind) -> std::span< val_t >;
    [[nodiscard]] inline auto getFieldView(size_t field_ind) const -> std::span< const val_t >;
    [[nodiscard]] auto        getNodeMap() const -> const node_map_t& { return m_node_to_dof_map; }
    template < size_t N >
    [[nodiscard]] auto getNodeValues(n_id_t node, std::span< const size_t, N > field_inds) const
        -> std::array< val_t, N >
        requires(N != std::dynamic_extent);

    void setField(size_t field_ind, val_t value) { std::ranges::fill(getFieldView(field_ind), value); }

private:
    size_t                     m_n_nodes, m_n_fields;
    std::unique_ptr< val_t[] > m_nodal_values;
    node_map_t                 m_node_to_dof_map;
};

SolutionManager::SolutionManager(const MeshPartition& mesh, size_t n_fields, val_t initial)
    : m_n_nodes{mesh.getAllNodes().size()},
      m_n_fields{n_fields},
      m_nodal_values{std::make_unique_for_overwrite< val_t[] >(m_n_nodes * m_n_fields)},
      m_node_to_dof_map(m_n_nodes)
{
    std::fill_n(m_nodal_values.get(), m_n_nodes * m_n_fields, initial);
    for (ptrdiff_t i = 0; auto node : mesh.getAllNodes())
        m_node_to_dof_map.emplace(node, i++);
}

template < size_t N >
[[nodiscard]] auto SolutionManager::getNodeValues(n_id_t node, std::span< const size_t, N > field_inds) const
    -> std::array< val_t, N >
    requires(N != std::dynamic_extent)
{
    const auto local_node_ind = m_node_to_dof_map.at(node);
    auto       retval         = std::array< val_t, N >{};
    std::ranges::transform(field_inds, retval.begin(), [&](auto i) { return getFieldView(i)[local_node_ind]; });
    return retval;
}

auto SolutionManager::getFieldView(size_t field_ind) const -> std::span< const val_t >
{
    return {std::next(m_nodal_values.get(), static_cast< ptrdiff_t >(field_ind * m_n_nodes)), m_n_nodes};
}

auto SolutionManager::getFieldView(size_t field_ind) -> std::span< val_t >
{
    return {std::next(m_nodal_values.get(), static_cast< ptrdiff_t >(field_ind * m_n_nodes)), m_n_nodes};
}
} // namespace lstr
#endif // L3STER_SOLUTIONMANAGER_HPP
