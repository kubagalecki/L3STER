#ifndef L3STER_POST_SOLUTIONMANAGER_HPP
#define L3STER_POST_SOLUTIONMANAGER_HPP

#include "l3ster/algsys/ComputeValuesAtNodes.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/EigenUtils.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr
{
class SolutionManager
{
public:
    template < el_o_t... orders >
    inline SolutionManager(const mesh::MeshPartition< orders... >& mesh, size_t n_fields, val_t initial = 0.);

    auto        nFields() const -> size_t { return m_n_fields; }
    auto        nNodes() const -> size_t { return m_n_nodes; }
    inline auto getFieldView(size_t field_ind) -> std::span< val_t >;
    inline auto getFieldView(size_t field_ind) const -> std::span< const val_t >;
    inline auto getRawView() -> Kokkos::View< val_t**, Kokkos::LayoutLeft >;
    inline auto getRawView() const -> Kokkos::View< const val_t**, Kokkos::LayoutLeft >;

    template < el_o_t... orders, ResidualKernel_c Kernel, size_t n_fields = 0 >
    void        setFields(const MpiComm&                          comm,
                          const mesh::MeshPartition< orders... >& mesh,
                          const Kernel&                           kernel,
                          const util::ArrayOwner< d_id_t >&       domain_ids,
                          const util::ArrayOwner< size_t >&       inds,
                          const post::FieldAccess< n_fields >&    field_access = {},
                          val_t                                   time         = 0.);
    inline void setFields(const util::ArrayOwner< size_t >& field_inds, val_t value);

    template < std::integral Index, size_t N >
    [[nodiscard]] auto getFieldAccess(const std::array< Index, N >& indices) const -> post::FieldAccess< N >;
    template < std::integral Index, size_t N >
    [[deprecated]] [[nodiscard]] auto makeFieldValueGetter(const std::array< Index, N >& indices) const
        -> post::FieldAccess< N >;
    template < size_t n_fields, RangeOfConvertibleTo_c< size_t > Indices >
    [[deprecated]] [[nodiscard]] auto makeFieldValueGetter(Indices&& indices) const -> post::FieldAccess< n_fields >;

private:
    template < size_t N >
    auto getNodeValuesGlobal(n_id_t node, const std::array< size_t, N >& field_inds) const -> std::array< val_t, N >;
    template < size_t N >
    auto getNodeValuesLocal(n_loc_id_t node, const std::array< size_t, N >& field_inds) const -> std::array< val_t, N >;

    size_t                                                      m_n_nodes, m_n_fields;
    std::unique_ptr< val_t[] >                                  m_nodal_values;
    std::shared_ptr< const util::SegmentedOwnership< n_id_t > > m_node_ownership;
};

template < el_o_t... orders >
SolutionManager::SolutionManager(const mesh::MeshPartition< orders... >& mesh, size_t n_fields, val_t initial)
    : m_n_nodes{mesh.getNNodes()},
      m_n_fields{n_fields},
      m_nodal_values{std::make_unique_for_overwrite< val_t[] >(m_n_nodes * m_n_fields)},
      m_node_ownership{mesh.getNodeOwnershipSharedPtr()}
{
    std::fill_n(m_nodal_values.get(), m_n_nodes * m_n_fields, initial);
}

template < size_t N >
auto SolutionManager::getNodeValuesGlobal(n_id_t node, const std::array< size_t, N >& field_inds) const
    -> std::array< val_t, N >
{
    const auto local_node_ind = m_node_ownership->getLocalIndex(node);
    auto       retval         = std::array< val_t, N >{};
    std::ranges::transform(field_inds, retval.begin(), [&](auto i) { return getFieldView(i)[local_node_ind]; });
    return retval;
}

template < size_t N >
auto SolutionManager::getNodeValuesLocal(n_loc_id_t node, const std::array< size_t, N >& field_inds) const
    -> std::array< val_t, N >
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

void SolutionManager::setFields(const util::ArrayOwner< size_t >& field_inds, val_t value)
{
    for (auto i : field_inds)
        std::ranges::fill(getFieldView(i), value);
}

template < el_o_t... orders, ResidualKernel_c Kernel, size_t n_fields >
void SolutionManager::setFields(const MpiComm&                          comm,
                                const mesh::MeshPartition< orders... >& mesh,
                                const Kernel&                           kernel,
                                const util::ArrayOwner< d_id_t >&       domain_ids,
                                const util::ArrayOwner< size_t >&       inds,
                                const post::FieldAccess< n_fields >&    field_access,
                                val_t                                   time)
{
    algsys::computeValuesAtNodes(comm, mesh, kernel, domain_ids, inds, getRawView(), field_access, time);
}

template < std::integral Index, size_t n_fields >
auto SolutionManager::getFieldAccess(const std::array< Index, n_fields >& field_inds) const
    -> post::FieldAccess< n_fields >
{
    auto field_inds_u32 = std::array< std::uint32_t, n_fields >{};
    std::ranges::transform(
        field_inds, field_inds_u32.begin(), [](auto i) { return util::exactIntegerCast< std::uint32_t >(i); });
    return {field_inds_u32, m_node_ownership, m_nodal_values.get()};
}

template < std::integral Index, size_t n_fields >
auto SolutionManager::makeFieldValueGetter(const std::array< Index, n_fields >& field_inds) const
    -> post::FieldAccess< n_fields >
{
    return getFieldAccess(field_inds);
}

template < size_t n_fields, RangeOfConvertibleTo_c< size_t > Indices >
auto SolutionManager::makeFieldValueGetter(Indices&& indices) const -> post::FieldAccess< n_fields >
{
    util::throwingAssert(std::ranges::distance(indices) == n_fields,
                         "The size of the passed index range differs from the value of the parameter");
    auto field_inds = std::array< std::uint32_t, n_fields >{};
    std::ranges::copy(std::forward< Indices >(indices), field_inds.begin());
    return getFieldAccess(field_inds);
}
} // namespace lstr
#endif // L3STER_POST_SOLUTIONMANAGER_HPP
