#ifndef L3STER_POST_SOLUTIONMANAGER_HPP
#define L3STER_POST_SOLUTIONMANAGER_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/EigenUtils.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <memory>

namespace lstr
{
class SolutionManager
{
public:
    using node_map_t = robin_hood::unordered_flat_map< n_id_t, ptrdiff_t >;

    template < el_o_t... orders >
    inline SolutionManager(const mesh::MeshPartition< orders... >& mesh, size_t n_fields, val_t initial = 0.);

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

    template < size_t n_fields >
    class FieldValueGetter
    {
        friend class SolutionManager;
        FieldValueGetter(const SolutionManager*                parent,
                         const std::array< size_t, n_fields >& field_inds = util::makeIotaArray< size_t, n_fields >())
            : m_parent{parent}, m_field_inds{field_inds}
        {}

    public:
        template < size_t n_nodes >
        auto operator()(const std::array< n_id_t, n_nodes >& nodes) const
            -> util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >;

    private:
        const SolutionManager*         m_parent;
        std::array< size_t, n_fields > m_field_inds;
    };

    template < size_t N >
    [[nodiscard]] auto makeFieldValueGetter(const std::array< size_t, N >& indices) const -> FieldValueGetter< N >;

private:
    size_t                     m_n_nodes, m_n_fields;
    std::unique_ptr< val_t[] > m_nodal_values;
    node_map_t                 m_node_to_dof_map;
};

template <>
class SolutionManager::FieldValueGetter< 0 >
{
public:
    template < size_t n_nodes >
    auto operator()(const std::array< n_id_t, n_nodes >&) const -> util::eigen::RowMajorMatrix< val_t, n_nodes, 0 >
    {
        return {};
    }
};
inline constexpr auto empty_field_val_getter = SolutionManager::FieldValueGetter< 0 >{};

template < size_t n_fields >
template < size_t n_nodes >
auto SolutionManager::FieldValueGetter< n_fields >::operator()(const std::array< n_id_t, n_nodes >& nodes) const
    -> util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >
{
    util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields > retval;
    for (size_t node_ind = 0; auto node : nodes)
    {
        const auto node_vals = m_parent->getNodeValues(node, std::span{m_field_inds});
        std::ranges::copy(node_vals, std::next(retval.data(), n_fields * node_ind++));
    }
    return retval;
}

template < el_o_t... orders >
SolutionManager::SolutionManager(const mesh::MeshPartition< orders... >& mesh, size_t n_fields, val_t initial)
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

template < size_t n_fields >
auto SolutionManager::makeFieldValueGetter(const std::array< size_t, n_fields >& field_inds) const
    -> FieldValueGetter< n_fields >
{
    return FieldValueGetter{this, field_inds};
}

/*
auto combineFieldValueGetters(auto&&... fval_getters)
{
    return [... getters = std::forward< decltype(fval_getters) >(fval_getters)]< size_t num_nodes >(
               const std::array< n_id_t, num_nodes >& nodes) {
        constexpr auto num_fields     = (decltype(std::invoke(getters, nodes))::ColsAtCompileTime + ...);
        auto           retval         = EigenRowMajorMatrix< val_t, num_nodes, num_fields >{};
        int            row            = 0;
        const auto     unconcatenated = std::make_tuple(std::invoke(getters, nodes)...);
        auto concatenator = [&, out = retval.data()]< size_t I >(std::integral_constant< size_t, I >) mutable {
            const auto&    vals   = std::get< I >(unconcatenated);
            constexpr auto n_cols = std::decay_t< decltype(vals) >::ColsAtCompileTime;
            for (int col = 0; col != n_cols; ++col)
                *out++ = vals(row, col);
        };
        for (size_t node = 0; node != num_nodes; ++node)
        {
            forConstexpr(concatenator, std::make_index_sequence< sizeof...(getters) >{});
            ++row;
        }
        return retval;
    };
}
 */
} // namespace lstr
#endif // L3STER_POST_SOLUTIONMANAGER_HPP
