#ifndef L3STER_POST_FIELDACCESS_HPP
#define L3STER_POST_FIELDACCESS_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/util/EigenUtils.hpp"
#include "l3ster/util/SegmentedOwnership.hpp"

namespace lstr::post
{
template < size_t num_fields >
class FieldAccess
{
public:
    FieldAccess() = default;
    FieldAccess(const std::array< std::uint32_t, num_fields >&              field_inds,
                std::shared_ptr< const util::SegmentedOwnership< n_id_t > > ownership,
                const val_t*                                                data) noexcept
        : m_field_inds{field_inds}, m_ownership{std::move(ownership)}, m_data{data}
    {}

    template < typename MatrixType, size_t num_nodes >
    void fill(MatrixType&& mat, const std::array< n_loc_id_t, num_nodes >& nodes) const
    {
        static_assert(std::remove_cvref_t< MatrixType >::RowsAtCompileTime == num_nodes);
        static_assert(std::remove_cvref_t< MatrixType >::ColsAtCompileTime == num_fields);
        const auto stride = m_ownership->localSize();
        for (auto&& [row, node] : nodes | std::views::enumerate)
            for (auto&& [col, field_ind] : m_field_inds | std::views::enumerate)
                mat(row, col) = m_data[node + field_ind * stride];
    }

    template < size_t n_nodes >
    auto getLocallyIndexed(const std::array< n_loc_id_t, n_nodes >& nodes) const
        -> util::eigen::RowMajorMatrix< val_t, n_nodes, num_fields >
    {
        auto retval = util::eigen::RowMajorMatrix< val_t, n_nodes, num_fields >{};
        fill(retval, nodes);
        return retval;
    }
    template < size_t n_nodes >
    auto getGloballyIndexed(const std::array< n_id_t, n_nodes >& nodes) const
        -> util::eigen::RowMajorMatrix< val_t, n_nodes, num_fields >
    {
        auto lids = std::array< n_loc_id_t, n_nodes >{};
        std::ranges::transform(nodes, lids.begin(), [&](n_id_t node) { return m_ownership->getLocalIndex(node); });
        return getLocallyIndexed(lids);
    }

private:
    std::array< std::uint32_t, num_fields >                     m_field_inds{};
    std::shared_ptr< const util::SegmentedOwnership< n_id_t > > m_ownership;
    const val_t*                                                m_data{};
};
static_assert(std::is_nothrow_move_constructible_v< FieldAccess< 1 > >);

template <>
class FieldAccess< 0 >
{
public:
    FieldAccess() = default;
    template < typename... Args >
    FieldAccess(const std::array< std::uint32_t, 0 >&,
                std::shared_ptr< const util::SegmentedOwnership< n_id_t > >,
                const val_t*) noexcept
    {}

    template < size_t n_nodes >
    auto getLocallyIndexed(const std::array< n_loc_id_t, n_nodes >&) const
        -> util::eigen::RowMajorMatrix< val_t, n_nodes, 0 >
    {
        return {};
    }
    template < size_t n_nodes >
    auto getGloballyIndexed(const std::array< n_id_t, n_nodes >&) const
        -> util::eigen::RowMajorMatrix< val_t, n_nodes, 0 >
    {
        return {};
    }
    template < typename MatrixType, size_t num_nodes >
    void fill(MatrixType&&, const std::array< n_loc_id_t, num_nodes >&) const
    {}
};
} // namespace lstr::post
#endif // L3STER_POST_FIELDACCESS_HPP
