#ifndef L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP
#define L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/util/TbbUtils.hpp"

#include <shared_mutex>

namespace lstr::detail
{
template < typename Derived >
class StaticCondensationManagerInterface
{
public:
    void beginAssembly() { static_cast< Derived* >(this)->beginAssemblyImpl(); }
    template < size_t max_dofs_per_node >
    void endAssembly(const MeshPartition&                             mesh,
                     const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                     tpetra_crsmatrix_t&                              global_matrix,
                     std::span< val_t >                               global_rhs)
    {
        static_cast< Derived* >(this)->endAssemblyImpl(mesh, node_dof_map, global_matrix, global_rhs);
    }
    template < ElementTypes ET, el_o_t EO, int system_size, size_t max_dofs_per_node, IndexRange_c auto field_inds >
    void condenseSystem(const NodeToLocalDofMap< max_dofs_per_node, 3 >&         node_dof_map,
                        tpetra_crsmatrix_t&                                      global_mat,
                        std::span< val_t >                                       global_rhs,
                        const eigen::RowMajorSquareMatrix< val_t, system_size >& local_matrix,
                        const Eigen::Vector< val_t, system_size >&               local_vector,
                        const Element< ET, EO >&                                 element,
                        ConstexprValue< field_inds >                             field_inds_ctwrpr)
    {
        static_cast< Derived* >(this)->condenseSystemImpl(
            node_dof_map, global_mat, global_rhs, local_matrix, local_vector, element, field_inds_ctwrpr);
    }
    template < size_t max_dofs_per_node >
    void recoverSolution(const MeshPartition&                             mesh,
                         const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                         std::span< const val_t >                         condensed_solution,
                         IndexRange_c auto&&                              sol_inds,
                         SolutionManager&                                 sol_man,
                         IndexRange_c auto&&                              sol_man_inds) const
    {
        static_cast< const Derived* >(this)->recoverSolutionImpl(mesh,
                                                                 node_dof_map,
                                                                 condensed_solution,
                                                                 std::forward< decltype(sol_inds) >(sol_inds),
                                                                 sol_man,
                                                                 std::forward< decltype(sol_man_inds) >(sol_man_inds));
    }
};

template < size_t max_dofs_per_node >
void updateSolutionPrimaryDofs(const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                               std::span< const val_t >                         condensed_solution,
                               const IndexRange_c auto&                         sol_inds,
                               SolutionManager&                                 sol_man,
                               IndexRange_c auto&&                              sol_man_inds)
{
    util::throwingAssert(std::ranges::distance(sol_man_inds) == std::ranges::distance(sol_inds),
                         "Source and destination indices length must match");
    util::throwingAssert(std::ranges::none_of(sol_inds, [&](size_t i) { return i >= max_dofs_per_node; }),
                         "Source index out of bounds");
    util::throwingAssert(std::ranges::none_of(sol_man_inds, [&](size_t i) { return i >= sol_man.nFields(); }),
                         "Destination index out of bounds");

    const auto dest_col_views = std::invoke([&] {
        std::vector< std::span< val_t > > retval;
        retval.reserve(std::ranges::distance(sol_man_inds));
        std::ranges::transform(
            sol_man_inds, std::back_inserter(retval), [&](size_t i) { return sol_man.getFieldView(i); });
        return retval;
    });

    auto iter_cache  = std::map{std::make_pair(size_t{}, node_dof_map.begin()),
                               std::make_pair(node_dof_map.size(), node_dof_map.end())};
    auto cache_mutex = std::shared_mutex{};

    const auto compute_begin_from_cache = [&](size_t index) {
        const auto index_requested = index;
        auto       it              = decltype(node_dof_map.begin()){};
        {
            const auto lock    = std::shared_lock{cache_mutex};
            const auto closest = std::prev(std::as_const(iter_cache).upper_bound(index));
            index              = closest->first;
            it                 = closest->second;
        }
        while (index != index_requested)
        {
            ++index;
            ++it;
        }
        return it;
    };
    const auto process_range = [&](auto begin, size_t range_size) {
        for (; range_size != 0; --range_size)
        {
            const auto [node, dofs]   = *begin++;
            const auto local_node_ind = sol_man.getNodeMap().at(node);
            for (size_t i = 0; size_t sol_ind : sol_inds)
            {
                const auto dof = dofs.back()[sol_ind];
                if (dof != NodeToLocalDofMap< max_dofs_per_node, 3 >::invalid_dof)
                    dest_col_views[i][local_node_ind] = condensed_solution[dof];
                ++i;
            }
        }
        return begin;
    };
    const auto cache_iter = [&](size_t index, auto it) {
        const auto lock   = std::lock_guard{cache_mutex};
        iter_cache[index] = it;
    };

    oneapi::tbb::parallel_for(oneapi::tbb::blocked_range< size_t >{0, node_dof_map.size(), 1 << 12},
                              [&](const oneapi::tbb::blocked_range< size_t >& index_range) {
                                  const auto begin = compute_begin_from_cache(index_range.begin());
                                  const auto end   = process_range(begin, index_range.size());
                                  cache_iter(index_range.end(), end);
                              });
}

template < CondensationPolicy CP >
class StaticCondensationManager;

template <>
class StaticCondensationManager< CondensationPolicy::None > :
    public StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::None > >
{
public:
    StaticCondensationManager() = default;
    template < size_t max_dofs_per_node, ProblemDef_c auto problem_def >
    StaticCondensationManager(const MeshPartition&,
                              const NodeToLocalDofMap< max_dofs_per_node, 3 >&,
                              ConstexprValue< problem_def >)
    {}

    void beginAssemblyImpl() {}
    template < size_t max_dofs_per_node >
    void endAssemblyImpl(const MeshPartition&,
                         const NodeToLocalDofMap< max_dofs_per_node, 3 >&,
                         tpetra_crsmatrix_t&,
                         std::span< val_t >)
    {}
    template < ElementTypes ET, el_o_t EO, int system_size, size_t max_dofs_per_node, IndexRange_c auto field_inds >
    void condenseSystemImpl(const NodeToLocalDofMap< max_dofs_per_node, 3 >&         node_dof_map,
                            tpetra_crsmatrix_t&                                      global_mat,
                            std::span< val_t >                                       global_rhs,
                            const eigen::RowMajorSquareMatrix< val_t, system_size >& local_mat,
                            const Eigen::Vector< val_t, system_size >&               local_vec,
                            const Element< ET, EO >&                                 element,
                            ConstexprValue< field_inds >                             field_inds_ctwrpr)
    {
        const auto [row_dofs, col_dofs, rhs_dofs] =
            detail::getUnsortedPrimaryDofs(element, node_dof_map, no_condensation, field_inds_ctwrpr);
        detail::scatterLocalSystem(local_mat, local_vec, global_mat, global_rhs, row_dofs, col_dofs, rhs_dofs);
    }
    template < size_t max_dofs_per_node >
    void recoverSolutionImpl(const MeshPartition&,
                             const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                             std::span< const val_t >                         condensed_solution,
                             IndexRange_c auto&&                              sol_inds,
                             SolutionManager&                                 sol_man,
                             IndexRange_c auto&&                              sol_man_inds) const
    {
        updateSolutionPrimaryDofs(node_dof_map, condensed_solution, sol_inds, sol_man, sol_man_inds);
    }
};

template <>
class StaticCondensationManager< CondensationPolicy::ElementBoundary > :
    public StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::ElementBoundary > >
{
public:
    StaticCondensationManager() = default;
    template < size_t max_dofs_per_node, ProblemDef_c auto problem_def >
    StaticCondensationManager(const MeshPartition&                             mesh,
                              const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                              ConstexprValue< problem_def >);

    inline void beginAssemblyImpl();
    template < size_t max_dofs_per_node >
    void endAssemblyImpl(const MeshPartition&                             mesh,
                         const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                         tpetra_crsmatrix_t&                              matrix,
                         std::span< val_t >                               rhs);
    template < ElementTypes ET, el_o_t EO, int system_size, size_t max_dofs_per_node, IndexRange_c auto field_inds >
    void condenseSystemImpl(const NodeToLocalDofMap< max_dofs_per_node, 3 >&         node_dof_map,
                            tpetra_crsmatrix_t&                                      global_mat,
                            std::span< val_t >                                       global_rhs,
                            const eigen::RowMajorSquareMatrix< val_t, system_size >& local_mat,
                            const Eigen::Vector< val_t, system_size >&               local_vec,
                            const Element< ET, EO >&                                 element,
                            ConstexprValue< field_inds >                             field_inds_ctwrpr);
    template < size_t max_dofs_per_node >
    void recoverSolutionImpl(const MeshPartition&,
                             const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                             std::span< const val_t >                         condensed_solution,
                             IndexRange_c auto&&                              sol_inds,
                             SolutionManager&                                 sol_man,
                             IndexRange_c auto&&                              sol_man_inds) const;

private:
    template < ElementTypes ET, el_o_t EO, size_t dofs_per_node >
    struct LocalDofInds
    {
        static constexpr size_t primary_dofs =
            dofs_per_node * ElementTraits< Element< ET, EO > >::boundary_node_inds.size();
        static constexpr size_t condensed_dofs =
            dofs_per_node * ElementTraits< Element< ET, EO > >::internal_node_inds.size();

        std::array< std::uint32_t, primary_dofs >   primary_src_inds, primary_dest_inds;
        std::array< std::uint32_t, condensed_dofs > cond_src_inds, cond_dest_inds;
    };

    template < ElementTypes ET, el_o_t EO, size_t max_dofs_per_node, ArrayOf_c< size_t > auto field_inds >
    static auto computeLocalDofInds(const Element< ET, EO >&                         element,
                                    const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                                    std::span< const std::uint8_t >                  internal_dof_inds,
                                    ConstexprValue< field_inds >                     field_inds_ctwrpr)
        -> LocalDofInds< ET, EO, field_inds.size() >;

    struct ElementCondData
    {
        std::span< const std::uint8_t >                         internal_dof_inds;
        eigen::DynamicallySizedMatrix< val_t, Eigen::RowMajor > diag_block, upper_block;
        Eigen::Vector< val_t, Eigen::Dynamic >                  rhs;
    };

    robin_hood::unordered_flat_map< el_id_t, ElementCondData > m_elem_data_map;
    std::vector< std::uint8_t >                                m_internal_dof_inds;
};

template < size_t max_dofs_per_node, ProblemDef_c auto problem_def >
StaticCondensationManager< CondensationPolicy::ElementBoundary >::StaticCondensationManager(
    const MeshPartition& mesh, const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map, ConstexprValue< problem_def >)
{
    const auto n_active_elems =
        std::transform_reduce(problem_def.begin(), problem_def.end(), size_t{}, std::plus{}, [&mesh](const auto& p) {
            return mesh.getDomain(p.first).getNElements();
        });
    m_internal_dof_inds.reserve(n_active_elems * max_dofs_per_node);
    const auto compute_elem_dof_info = [&dof_map]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
        // Bitmap is initially inverted, i.e., 0 implies that the dof is active (avoids awkward all-true construction)
        using dof_bmp_t    = std::bitset< max_dofs_per_node >;
        auto dof_bmp_range = getBoundaryNodes(element) | std::views::transform(dof_map) | std::views::keys |
                             std::views::transform([&](const std::array< local_dof_t, max_dofs_per_node >& dofs) {
                                 auto retval = dof_bmp_t{};
                                 for (size_t i = 0; auto dof : dofs)
                                 {
                                     retval[i] = dof == NodeToLocalDofMap< max_dofs_per_node, 3 >::invalid_dof;
                                     ++i;
                                 }
                                 return retval;
                             }) |
                             std::views::common;
        const auto [n_boundary_dofs, internal_dof_bmp] = std::transform_reduce(
            std::ranges::cbegin(dof_bmp_range),
            std::ranges::cend(dof_bmp_range),
            std::make_pair(size_t{0}, dof_bmp_t{}),
            [](const auto& p1, const auto& p2) { return std::make_pair(p1.first + p2.first, p1.second | p2.second); },
            [](const dof_bmp_t& dof_bmp) { return std::make_pair(dof_bmp.count(), dof_bmp); });
        return std::make_pair(n_boundary_dofs, ~internal_dof_bmp);
    };
    mesh.visit(
        [&compute_elem_dof_info, this]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            const auto [n_boundary_dofs, internal_dof_bmp] = compute_elem_dof_info(element);
            const auto el_int_dofs_begin                   = m_internal_dof_inds.end();
            for (std::uint8_t i = 0; i != internal_dof_bmp.size(); ++i)
                if (internal_dof_bmp[i])
                    m_internal_dof_inds.push_back(i);
            const auto el_int_dofs_end = m_internal_dof_inds.end();
            const auto n_internal_dofs =
                ElementTraits< Element< ET, EO > >::internal_node_inds.size() * internal_dof_bmp.count();
            m_elem_data_map.emplace(element.getId(),
                                    ElementCondData{.internal_dof_inds{el_int_dofs_begin, el_int_dofs_end},
                                                    .diag_block{n_internal_dofs, n_internal_dofs},
                                                    .upper_block{n_boundary_dofs, n_internal_dofs},
                                                    .rhs{n_internal_dofs, 1}});
        },
        problem_def | std::views::transform([](const auto& pair) { return pair.first; }),
        std::execution::seq);
    m_internal_dof_inds.shrink_to_fit();
}

void StaticCondensationManager< CondensationPolicy::ElementBoundary >::beginAssemblyImpl()
{
    for (auto& [id, payload] : m_elem_data_map)
    {
        payload.diag_block.setZero();
        payload.upper_block.setZero();
        payload.rhs.setZero();
    }
}

template < size_t max_dofs_per_node >
void StaticCondensationManager< CondensationPolicy::ElementBoundary >::recoverSolutionImpl(
    [[maybe_unused]] const MeshPartition&                             mesh,
    [[maybe_unused]] const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
    [[maybe_unused]] std::span< const val_t >                         condensed_solution,
    [[maybe_unused]] IndexRange_c auto&&                              sol_inds,
    [[maybe_unused]] SolutionManager&                                 sol_man,
    [[maybe_unused]] IndexRange_c auto&&                              sol_man_inds) const
{}

template < ElementTypes ET, el_o_t EO, int system_size, size_t max_dofs_per_node, IndexRange_c auto field_inds >
void StaticCondensationManager< CondensationPolicy::ElementBoundary >::condenseSystemImpl(
    const NodeToLocalDofMap< max_dofs_per_node, 3 >&               node_dof_map,
    tpetra_crsmatrix_t&                                            global_mat,
    std::span< val_t >                                             global_rhs,
    const eigen::RowMajorSquareMatrix< lstr::val_t, system_size >& local_mat,
    const Eigen::Vector< lstr::val_t, system_size >&               local_vec,
    const Element< ET, EO >&                                       element,
    ConstexprValue< field_inds >                                   field_inds_ctwrpr)
{
    auto&      cond_data = m_elem_data_map.at(element.getId());
    const auto dof_inds  = computeLocalDofInds(element, node_dof_map, cond_data.internal_dof_inds, field_inds_ctwrpr);

    // Primary diagonal block + RHS
    constexpr int n_primary_dofs  = std::tuple_size_v< decltype(dof_inds.primary_src_inds) >;
    auto          primary_upd_mat = eigen::RowMajorSquareMatrix< val_t, n_primary_dofs >{};
    auto          primary_upd_rhs = Eigen::Vector< val_t, n_primary_dofs >{};
    for (size_t row_ind = 0; auto src_row : dof_inds.primary_src_inds)
    {
        for (size_t col_ind = 0; auto src_col : dof_inds.primary_src_inds)
            primary_upd_mat(row_ind, col_ind++) = local_mat(src_row, src_col);
        primary_upd_rhs[row_ind++] = local_vec[src_row];
    }
    const auto [row_dofs, col_dofs, rhs_dofs] =
        detail::getUnsortedPrimaryDofs(element, node_dof_map, element_boundary, field_inds_ctwrpr);
    detail::scatterLocalSystem(primary_upd_mat, primary_upd_rhs, global_mat, global_rhs, row_dofs, col_dofs, rhs_dofs);

    // Condensed diagonal block + RHS
    for (size_t row_ind = 0; auto src_row : dof_inds.cond_src_inds)
    {
        const auto dest_row = dof_inds.cond_dest_inds[row_ind++];
        for (size_t col_ind = 0; auto src_col : dof_inds.cond_src_inds)
        {
            const auto dest_col = dof_inds.cond_dest_inds[col_ind++];
            cond_data.diag_block(dest_row, dest_col) += local_mat(src_row, src_col);
        }
        cond_data.rhs[dest_row] += local_vec[src_row];
    }

    // Off-diagonal block (upper)
    for (size_t row_ind = 0; auto src_row : dof_inds.primary_src_inds)
    {
        const auto dest_row = dof_inds.primary_dest_inds[row_ind++];
        for (size_t col_ind = 0; auto src_col : dof_inds.cond_src_inds)
        {
            const auto dest_col = dof_inds.cond_dest_inds[col_ind++];
            cond_data.upper_block(dest_row, dest_col) += local_mat(src_row, src_col);
        }
    }
}

template < size_t max_dofs_per_node >
void StaticCondensationManager< CondensationPolicy::ElementBoundary >::endAssemblyImpl(
    const MeshPartition&                             mesh,
    const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
    tpetra_crsmatrix_t&                              matrix,
    std::span< val_t >                               rhs)
{
    for (auto& [id, elem_data] : m_elem_data_map)
    {
        const auto elem_variant = mesh.find(id);
        std::visit(
            [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >* element_ptr) {
                {
                    eigen::DynamicallySizedMatrix< val_t, Eigen::RowMajor > diag_inv = elem_data.diag_block.inverse();
                    elem_data.diag_block                                             = std::move(diag_inv);
                }
                const eigen::DynamicallySizedMatrix< val_t, Eigen::RowMajor > primary_upd_mat =
                    -(elem_data.upper_block * elem_data.diag_block * elem_data.upper_block.transpose());
                const Eigen::Vector< val_t, Eigen::Dynamic > primary_upd_rhs =
                    -(elem_data.upper_block * elem_data.diag_block * elem_data.rhs);

                const auto [row_dofs, col_dofs, rhs_dofs] =
                    detail::getUnsortedPrimaryDofs(*element_ptr, dof_map, element_boundary);
                detail::scatterLocalSystem(primary_upd_mat, primary_upd_rhs, matrix, rhs, row_dofs, col_dofs, rhs_dofs);
            },
            elem_variant->first);
    }
}

template < ElementTypes ET, el_o_t EO, size_t max_dofs_per_node, ArrayOf_c< size_t > auto field_inds >
auto StaticCondensationManager< CondensationPolicy::ElementBoundary >::computeLocalDofInds(
    const Element< ET, EO >&                         element,
    const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
    std::span< const std::uint8_t >                  internal_dof_inds,
    ConstexprValue< field_inds >) -> LocalDofInds< ET, EO, field_inds.size() >
{
    constexpr auto inds_to_bmp = [](const auto& inds) {
        auto retval = std::array< bool, max_dofs_per_node >{};
        for (auto i : inds)
            retval[i] = true;
        return retval;
    };
    constexpr auto active_inds_bmp   = inds_to_bmp(field_inds);
    const auto     internal_dof_bmp  = inds_to_bmp(internal_dof_inds);
    auto           retval            = LocalDofInds< ET, EO, field_inds.size() >{};
    size_t         primary_write_ind = 0, cond_write_ind = 0;
    std::uint32_t  local_system_ind = 0, primary_ind = 0, cond_ind = 0;
    auto           boundary_ind_iter = ElementTraits< Element< ET, EO > >::boundary_node_inds.cbegin();
    for (el_locind_t node_ind = 0; auto node : element.getNodes())
    {
        if (*boundary_ind_iter == node_ind++)
        {
            ++boundary_ind_iter;
            const auto& node_dofs = dof_map(node).front();
            for (size_t i = 0; i != max_dofs_per_node; ++i)
            {
                if (active_inds_bmp[i])
                {
                    retval.primary_src_inds[primary_write_ind]    = local_system_ind++;
                    retval.primary_dest_inds[primary_write_ind++] = primary_ind++;
                }
                else if (node_dofs[i] != NodeToLocalDofMap< max_dofs_per_node, 3 >::invalid_dof)
                    ++primary_ind;
            }
        }
        else
        {
            for (size_t i = 0; i != max_dofs_per_node; ++i)
            {
                if (active_inds_bmp[i])
                {
                    retval.cond_src_inds[cond_write_ind]    = local_system_ind++;
                    retval.cond_dest_inds[cond_write_ind++] = cond_ind++;
                }
                else if (internal_dof_bmp[i])
                    ++cond_ind;
            }
        }
    }
    return retval;
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP
