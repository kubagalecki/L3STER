#ifndef L3STER_POST_SOLUTIONMANAGER_HPP
#define L3STER_POST_SOLUTIONMANAGER_HPP

#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/EigenUtils.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include "Kokkos_Core.hpp"
#include <memory>

namespace lstr
{
class SolutionManager
{
public:
    using node_map_t = robin_hood::unordered_flat_map< n_id_t, ptrdiff_t >;

    template < el_o_t... orders >
    inline SolutionManager(const mesh::MeshPartition< orders... >& mesh, size_t n_fields, val_t initial = 0.);

    auto        nFields() const -> size_t { return m_n_fields; }
    auto        nNodes() const -> size_t { return m_n_nodes; }
    inline auto getFieldView(size_t field_ind) -> std::span< val_t >;
    inline auto getFieldView(size_t field_ind) const -> std::span< const val_t >;
    auto        getNodeMap() const -> const node_map_t& { return m_node_to_dof_map; }
    inline auto getRawView() -> Kokkos::View< val_t**, Kokkos::LayoutLeft >;
    inline auto getRawView() const -> Kokkos::View< const val_t**, Kokkos::LayoutLeft >;
    template < size_t N >
    auto getNodeValuesGlobal(n_id_t node, const std::array< size_t, N >& field_inds) const -> std::array< val_t, N >;
    template < size_t N >
    auto getNodeValuesLocal(n_loc_id_t node, const std::array< size_t, N >& field_inds) const -> std::array< val_t, N >;

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
        auto getGloballyIndexed(const std::array< n_id_t, n_nodes >& nodes) const
            -> util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >;
        template < size_t n_nodes >
        auto getLocallyIndexed(const std::array< n_loc_id_t, n_nodes >& nodes) const
            -> util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >;

    private:
        const SolutionManager*         m_parent;
        std::array< size_t, n_fields > m_field_inds;
    };

    template < std::integral Index, size_t N >
    [[nodiscard]] auto makeFieldValueGetter(const std::array< Index, N >& indices) const -> FieldValueGetter< N >;
    template < size_t n_fields, RangeOfConvertibleTo_c< size_t > Indices >
    [[nodiscard]] auto makeFieldValueGetter(Indices&& indices) const -> FieldValueGetter< n_fields >;

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
    auto
    getGloballyIndexed(const std::array< n_id_t, n_nodes >&) const -> util::eigen::RowMajorMatrix< val_t, n_nodes, 0 >
    {
        return {};
    }
    template < size_t n_nodes >
    auto getLocallyIndexed(const std::array< n_loc_id_t, n_nodes >&) const
        -> util::eigen::RowMajorMatrix< val_t, n_nodes, 0 >
    {
        return {};
    }
};
inline constexpr auto empty_field_val_getter = SolutionManager::FieldValueGetter< 0 >{};

template < size_t n_fields >
template < size_t n_nodes >
auto SolutionManager::FieldValueGetter< n_fields >::getGloballyIndexed(const std::array< n_id_t, n_nodes >& nodes) const
    -> util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >
{
    util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields > retval;
    const auto                                              vals_from_node = [&](auto node) {
        return m_parent->getNodeValuesGlobal(node, m_field_inds);
    };
    std::ranges::copy(nodes | std::views::transform(vals_from_node) | std::views::join, retval.data());
    return retval;
}

template < size_t n_fields >
template < size_t n_nodes >
auto SolutionManager::FieldValueGetter< n_fields >::getLocallyIndexed(
    const std::array< n_loc_id_t, n_nodes >& nodes) const -> util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >
{
    util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields > retval;
    const auto                                              vals_from_node = [&](auto node) {
        return m_parent->getNodeValuesLocal(node, m_field_inds);
    };
    std::ranges::copy(nodes | std::views::transform(vals_from_node) | std::views::join, retval.data());
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
    for (auto&& [i, node] : mesh.getAllNodes() | std::views::enumerate)
        m_node_to_dof_map.emplace(node, i);
}

template < size_t N >
auto SolutionManager::getNodeValuesGlobal(n_id_t                         node,
                                          const std::array< size_t, N >& field_inds) const -> std::array< val_t, N >
{
    const auto local_node_ind = m_node_to_dof_map.at(node);
    auto       retval         = std::array< val_t, N >{};
    std::ranges::transform(field_inds, retval.begin(), [&](auto i) { return getFieldView(i)[local_node_ind]; });
    return retval;
}

template < size_t N >
auto SolutionManager::getNodeValuesLocal(n_loc_id_t                     node,
                                         const std::array< size_t, N >& field_inds) const -> std::array< val_t, N >
{
    auto retval = std::array< val_t, N >{};
    std::ranges::transform(field_inds, retval.begin(), [&](auto i) { return getFieldView(i)[node]; });
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

auto SolutionManager::getRawView() -> Kokkos::View< val_t**, Kokkos::LayoutLeft >
{
    return Kokkos::View< val_t**, Kokkos::LayoutLeft >{m_nodal_values.get(), m_n_nodes, m_n_fields};
}

auto SolutionManager::getRawView() const -> Kokkos::View< const val_t**, Kokkos::LayoutLeft >
{
    return Kokkos::View< const val_t**, Kokkos::LayoutLeft >{m_nodal_values.get(), m_n_nodes, m_n_fields};
}

template < std::integral Index, size_t n_fields >
auto SolutionManager::makeFieldValueGetter(const std::array< Index, n_fields >& field_inds) const
    -> FieldValueGetter< n_fields >
{
    auto field_inds_size_t = std::array< size_t, n_fields >{};
    std::ranges::transform(
        field_inds, field_inds_size_t.begin(), [](auto i) { return util::exactIntegerCast< size_t >(i); });
    return FieldValueGetter{this, field_inds_size_t};
}

template < size_t n_fields, RangeOfConvertibleTo_c< size_t > Indices >
auto SolutionManager::makeFieldValueGetter(Indices&& indices) const -> FieldValueGetter< n_fields >
{
    util::throwingAssert(std::ranges::distance(indices) == n_fields,
                         "The size of the passed index range differs from the value of the parameter");

    auto field_inds = std::array< size_t, n_fields >{};
    std::ranges::copy(std::forward< Indices >(indices), field_inds.begin());
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
