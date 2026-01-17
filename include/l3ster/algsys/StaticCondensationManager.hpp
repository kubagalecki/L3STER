#ifndef L3STER_ALGSYS_STATICCONDENSATIONMANAGER_HPP
#define L3STER_ALGSYS_STATICCONDENSATIONMANAGER_HPP

#include "l3ster/algsys/ScatterLocalSystem.hpp"
#include "l3ster/dofs/DofsFromNodes.hpp"
#include "l3ster/post/SolutionManager.hpp"
#include "l3ster/util/ScopeGuards.hpp"
#include "l3ster/util/TbbUtils.hpp"

namespace lstr::algsys
{
template < typename Derived >
class StaticCondensationManagerInterface
{
public:
    void beginAssembly() { static_cast< Derived* >(this)->beginAssemblyImpl(); }
    template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
    void endAssembly(const mesh::MeshPartition< orders... >&                mesh,
                     const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                     tpetra_crsmatrix_t&                                    global_matrix,
                     const std::array< std::span< val_t >, n_rhs >&         global_rhs)
    {
        static_cast< Derived* >(this)->endAssemblyImpl(mesh, node_dof_map, global_matrix, global_rhs);
    }
    template < mesh::ElementType ET,
               el_o_t            EO,
               int               system_size,
               size_t            n_rhs,
               size_t            max_dofs_per_node,
               IndexRange_c auto field_inds >
    void condenseSystem(const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >&         node_dof_map,
                        tpetra_crsmatrix_t&                                            global_mat,
                        const std::array< std::span< val_t >, n_rhs >&                 global_rhs,
                        const util::eigen::RowMajorSquareMatrix< val_t, system_size >& local_matrix,
                        const Eigen::Matrix< val_t, system_size, int{n_rhs} >&         local_rhs,
                        const mesh::Element< ET, EO >&                                 element,
                        util::ConstexprValue< field_inds >                             field_inds_ctwrpr)
    {
        static_cast< Derived* >(this)->condenseSystemImpl(
            node_dof_map, global_mat, global_rhs, local_matrix, local_rhs, element, field_inds_ctwrpr);
    }
    template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
    void recoverSolution(const mesh::MeshPartition< orders... >&                mesh,
                         const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                         const std::array< std::span< const val_t >, n_rhs >&   condensed_solutions,
                         const util::ArrayOwner< size_t >&                      sol_inds,
                         SolutionManager&                                       sol_man,
                         const util::ArrayOwner< size_t >&                      sol_man_inds) const
    {
        static_cast< const Derived* >(this)->recoverSolutionImpl(
            mesh, node_dof_map, condensed_solutions, sol_inds, sol_man, sol_man_inds);
    }

protected:
    StaticCondensationManagerInterface() = default;
    template < el_o_t... orders >
    StaticCondensationManagerInterface(const mesh::MeshPartition< orders... >& mesh)
        : m_node_ownership{mesh.getNodeOwnershipSharedPtr()}
    {}

    template < size_t max_dofs_per_node, size_t n_rhs >
    static void validateSolutionUpdateInds(const util::ArrayOwner< size_t >& sol_inds,
                                           SolutionManager&                  sol_man,
                                           const util::ArrayOwner< size_t >& sol_man_inds);
    template < size_t max_dofs_per_node, size_t n_rhs >
    void updateSolutionPrimaryDofs(const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                                   const std::array< std::span< const val_t >, n_rhs >&   condensed_solutions,
                                   const util::ArrayOwner< size_t >&                      sol_inds,
                                   SolutionManager&                                       sol_man,
                                   const util::ArrayOwner< size_t >&                      sol_man_inds) const;

    auto getNodeLid(n_id_t node) const { return static_cast< n_loc_id_t >(m_node_ownership->getLocalIndex(node)); }

private:
    std::shared_ptr< const util::SegmentedOwnership< n_id_t > > m_node_ownership;
};

template < CondensationPolicy CP >
class StaticCondensationManager;

template <>
class StaticCondensationManager< CondensationPolicy::None > :
    public StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::None > >
{
    using Base = StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::None > >;

public:
    StaticCondensationManager() = default;
    template < el_o_t... orders, size_t max_dofs_per_node >
    StaticCondensationManager(const mesh::MeshPartition< orders... >& mesh,
                              const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >&,
                              const ProblemDefinition< max_dofs_per_node >&,
                              size_t)
        : Base{mesh}
    {}

    void beginAssemblyImpl() {}
    template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
    void endAssemblyImpl(const mesh::MeshPartition< orders... >&,
                         const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >&,
                         tpetra_crsmatrix_t&,
                         const std::array< std::span< val_t >, n_rhs >&)
    {}
    template < mesh::ElementType ET,
               el_o_t            EO,
               size_t            n_rhs,
               int               system_size,
               size_t            max_dofs_per_node,
               IndexRange_c auto field_inds >
    void condenseSystemImpl(const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >&         node_dof_map,
                            tpetra_crsmatrix_t&                                            global_mat,
                            const std::array< std::span< val_t >, n_rhs >&                 global_rhs,
                            const util::eigen::RowMajorSquareMatrix< val_t, system_size >& local_mat,
                            const Eigen::Matrix< val_t, system_size, int{n_rhs} >&         local_rhs,
                            const mesh::Element< ET, EO >&                                 element,
                            util::ConstexprValue< field_inds >                             field_inds_ctwrpr)
    {
        const auto [row_dofs, col_dofs, rhs_dofs] =
            dofs::getDofsFromNodes(element.nodes, node_dof_map, field_inds_ctwrpr);
        scatterLocalSystem(local_mat, local_rhs, global_mat, global_rhs, row_dofs, col_dofs, rhs_dofs);
    }
    template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
    void recoverSolutionImpl(const mesh::MeshPartition< orders... >&,
                             const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                             const std::array< std::span< const val_t >, n_rhs >&   condensed_solutions,
                             const util::ArrayOwner< size_t >&                      sol_inds,
                             SolutionManager&                                       sol_man,
                             const util::ArrayOwner< size_t >&                      sol_man_inds) const
    {
        validateSolutionUpdateInds< max_dofs_per_node, n_rhs >(sol_inds, sol_man, sol_man_inds);
        updateSolutionPrimaryDofs(node_dof_map, condensed_solutions, sol_inds, sol_man, sol_man_inds);
    }
};

template <>
class StaticCondensationManager< CondensationPolicy::ElementBoundary > :
    public StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::ElementBoundary > >
{
    using Base = StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::ElementBoundary > >;
    template < mesh::ElementType ET, el_o_t EO, size_t dofs_per_node >
    struct LocalDofInds
    {
        static constexpr size_t n_primary_dofs =
            dofs_per_node * mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds.size();
        static constexpr size_t n_condensed_dofs =
            dofs_per_node * mesh::ElementTraits< mesh::Element< ET, EO > >::internal_node_inds.size();

        std::array< std::uint32_t, n_primary_dofs >   primary_src_inds, primary_dest_inds;
        std::array< std::uint32_t, n_condensed_dofs > cond_src_inds, cond_dest_inds;
    };

    struct ElementCondData
    {
        size_t                                                        internal_dofs_offs, internal_dofs_size;
        util::eigen::DynamicallySizedMatrix< val_t, Eigen::RowMajor > diag_block, diag_block_inv, upper_block;
        Eigen::Matrix< val_t, Eigen::Dynamic, Eigen::Dynamic >        rhs;
    };

public:
    StaticCondensationManager() = default;
    template < el_o_t... orders, size_t max_dofs_per_node >
    StaticCondensationManager(const mesh::MeshPartition< orders... >&                mesh,
                              const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                              const ProblemDefinition< max_dofs_per_node >&          problem_def,
                              size_t                                                 n_rhs);

    inline void beginAssemblyImpl();
    template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
    void endAssemblyImpl(const mesh::MeshPartition< orders... >&                mesh,
                         const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                         tpetra_crsmatrix_t&                                    matrix,
                         const std::array< std::span< val_t >, n_rhs >&         rhs);
    template < mesh::ElementType ET,
               el_o_t            EO,
               size_t            n_rhs,
               int               system_size,
               size_t            max_dofs_per_node,
               IndexRange_c auto field_inds >
    void condenseSystemImpl(const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >&         node_dof_map,
                            tpetra_crsmatrix_t&                                            global_mat,
                            const std::array< std::span< val_t >, n_rhs >&                 global_rhs,
                            const util::eigen::RowMajorSquareMatrix< val_t, system_size >& local_mat,
                            const Eigen::Matrix< val_t, system_size, int{n_rhs} >&         local_rhs,
                            const mesh::Element< ET, EO >&                                 element,
                            util::ConstexprValue< field_inds >                             field_inds_ctwrpr);
    template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
    void recoverSolutionImpl(const mesh::MeshPartition< orders... >&,
                             const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                             const std::array< std::span< const val_t >, n_rhs >&   condensed_solutions,
                             const util::ArrayOwner< size_t >&                      sol_inds,
                             SolutionManager&                                       sol_man,
                             const util::ArrayOwner< size_t >&                      sol_man_inds) const;

private:
    inline auto getInternalDofInds(const ElementCondData& cond_data) const -> std::span< const std::uint8_t >;

    template < mesh::ElementType ET, el_o_t EO, size_t max_dofs_per_node, ArrayOf_c< size_t > auto field_inds >
    auto computeLocalDofInds(const mesh::Element< ET, EO >&                         element,
                             const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                             const ElementCondData&                                 cond_data,
                             util::ConstexprValue< field_inds >                     field_inds_ctwrpr)
        -> LocalDofInds< ET, EO, field_inds.size() >;

    robin_hood::unordered_flat_map< el_id_t, ElementCondData > m_elem_data_map;
    std::vector< std::uint8_t >                                m_internal_dof_inds;
    std::vector< el_id_t >                                     m_element_ids;
};

template < typename Derived >
template < size_t max_dofs_per_node, size_t n_rhs >
void StaticCondensationManagerInterface< Derived >::validateSolutionUpdateInds(
    const util::ArrayOwner< size_t >& sol_inds,
    SolutionManager&                  sol_man,
    const util::ArrayOwner< size_t >& sol_man_inds)
{
    util::throwingAssert(std::ranges::distance(sol_man_inds) == std::ranges::distance(sol_inds) * ptrdiff_t{n_rhs},
                         "The number of solution manager indices (target) must be equal to the number of DOF indices "
                         "multiplied by the number of RHS/solution vectors (source)");
    util::throwingAssert(std::ranges::all_of(sol_inds, [&](size_t i) { return i < max_dofs_per_node; }),
                         "The DOF (source) indices must be in the range [ 0, #DOFs )");
    util::throwingAssert(
        std::ranges::all_of(sol_man_inds, [&](size_t i) { return i < sol_man.nFields(); }),
        "The solution manager (target) indices must be smaller than the number of allocated solutions");
}

template < typename Derived >
template < size_t max_dofs_per_node, size_t n_rhs >
void StaticCondensationManagerInterface< Derived >::updateSolutionPrimaryDofs(
    const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
    const std::array< std::span< const val_t >, n_rhs >&   condensed_solutions,
    const util::ArrayOwner< size_t >&                      sol_inds,
    SolutionManager&                                       sol_man,
    const util::ArrayOwner< size_t >&                      sol_man_inds) const
{
    const auto dest_col_views      = std::invoke([&] {
        util::ArrayOwner< std::span< val_t > > retval(sol_man_inds.size());
        std::ranges::transform(sol_man_inds, retval.begin(), [&](size_t i) { return sol_man.getFieldView(i); });
        return retval;
    });
    const auto update_node_entries = [&](const auto& map_entry) {
        const auto& [node, dof_triplet] = map_entry;
        const auto& dofs                = dof_triplet.back();
        const auto  local_node_ind      = getNodeLid(node);
        for (size_t target_ind = 0; auto src_ind : sol_inds)
        {
            const auto local_dof = dofs[src_ind];
            for (size_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
            {
                const auto src_sol_span = condensed_solutions[rhs_ind];
                const auto dest_span    = dest_col_views[target_ind++];
                if (dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >::isValid(local_dof))
                    dest_span[local_node_ind] = src_sol_span[local_dof];
            }
        }
    };
    util::tbb::parallelFor(node_dof_map, update_node_entries);
}

template < el_o_t... orders, size_t max_dofs_per_node >
StaticCondensationManager< CondensationPolicy::ElementBoundary >::StaticCondensationManager(
    const mesh::MeshPartition< orders... >&                mesh,
    const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
    const ProblemDefinition< max_dofs_per_node >&          problem_def,
    size_t                                                 n_rhs)
    : Base{mesh}
{
    const auto compute_elem_dof_info = [&dof_map]< mesh::ElementType ET, el_o_t EO >(
                                           const mesh::Element< ET, EO >& element) {
        // Bitmap is initially inverted, i.e., 0 implies that the dof is active (avoids awkward all-true construction)
        using dof_bmp_t                = std::bitset< max_dofs_per_node >;
        const auto get_invalid_dof_bmp = [&](const std::array< local_dof_t, max_dofs_per_node >& dofs) {
            auto retval = dof_bmp_t{};
            for (auto&& [i, dof] : dofs | std::views::enumerate)
                retval[i] = dof == invalid_local_dof;
            return retval;
        };
        constexpr auto add_unset = [](const dof_bmp_t& dof_bmp) {
            return std::make_pair(dof_bmp.size() - dof_bmp.count(), dof_bmp);
        };
        auto dof_bmp_range = getBoundaryNodes(element) | std::views::transform(std::cref(dof_map)) | std::views::keys |
                             std::views::transform(get_invalid_dof_bmp) | std::views::transform(add_unset);
        const auto [n_boundary_dofs, internal_dof_bmp] =
            std::ranges::fold_left(dof_bmp_range, std::make_pair(0uz, dof_bmp_t{}), [](const auto& p1, const auto& p2) {
                return std::make_pair(p1.first + p2.first, p1.second | p2.second);
            });
        return std::make_pair(n_boundary_dofs, ~internal_dof_bmp);
    };
    const auto insert_elem_info = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
        const auto [n_boundary_dofs, internal_dof_bmp] = compute_elem_dof_info(element);
        const auto el_int_dofs_offs                    = m_internal_dof_inds.size();
        const auto n_int_dofs_per_node                 = internal_dof_bmp.count();
        for (std::uint8_t i = 0; i != internal_dof_bmp.size(); ++i)
            if (internal_dof_bmp[i])
                m_internal_dof_inds.push_back(i);
        constexpr auto num_internal_nodes = mesh::ElementTraits< mesh::Element< ET, EO > >::internal_node_inds.size();
        const auto     n_internal_dofs    = num_internal_nodes * n_int_dofs_per_node;
        m_elem_data_map.emplace(element.id,
                                ElementCondData{.internal_dofs_offs{el_int_dofs_offs},
                                                .internal_dofs_size{n_int_dofs_per_node},
                                                .diag_block{n_internal_dofs, n_internal_dofs},
                                                .diag_block_inv{n_internal_dofs, n_internal_dofs},
                                                .upper_block{n_boundary_dofs, n_internal_dofs},
                                                .rhs{n_internal_dofs, n_rhs}});
    };
    mesh.visit(insert_elem_info, problem_def.getDomains(), std::execution::seq);
    m_internal_dof_inds.shrink_to_fit();
    m_element_ids.reserve(m_elem_data_map.size());
    std::ranges::transform(m_elem_data_map, std::back_inserter(m_element_ids), [](const auto& p) { return p.first; });
}

void StaticCondensationManager< CondensationPolicy::ElementBoundary >::beginAssemblyImpl()
{
    util::tbb::parallelFor(m_element_ids, [&](el_id_t id) {
        auto& payload = m_elem_data_map.at(id);
        payload.diag_block.setZero();
        payload.upper_block.setZero();
        payload.rhs.setZero();
    });
}

template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
void StaticCondensationManager< CondensationPolicy::ElementBoundary >::endAssemblyImpl(
    const mesh::MeshPartition< orders... >&                mesh,
    const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
    tpetra_crsmatrix_t&                                    matrix,
    const std::array< std::span< val_t >, n_rhs >&         rhs)
{
    const auto n_cores       = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNCores();
    const auto max_par_guard = util::MaxParallelismGuard{n_cores};
    util::tbb::parallelFor(m_element_ids, [&](el_id_t id) {
        auto&      elem_data           = m_elem_data_map.at(id);
        const auto element_ptr_variant = mesh.find(id).value();
        const auto finalize_element    = [&]< mesh::ElementType ET, el_o_t EO >(
                                          const mesh::Element< ET, EO >* element_ptr) {
            elem_data.diag_block_inv = elem_data.diag_block.inverse();
            thread_local util::eigen::DynamicallySizedMatrix< val_t, Eigen::RowMajor > primary_upd_mat;
            thread_local Eigen::Matrix< val_t, Eigen::Dynamic, int{n_rhs} >            primary_upd_rhs;
            primary_upd_mat = -(elem_data.upper_block * elem_data.diag_block_inv * elem_data.upper_block.transpose());
            primary_upd_rhs = -(elem_data.upper_block * elem_data.diag_block_inv * elem_data.rhs);
            const auto [row_dofs, col_dofs, rhs_dofs] = dofs::getDofsFromNodes(
                dofs::getPrimaryNodesArray< CondensationPolicy::ElementBoundary >(*element_ptr), dof_map);
            scatterLocalSystem(primary_upd_mat, primary_upd_rhs, matrix, rhs, row_dofs, col_dofs, rhs_dofs);
        };
        std::visit(finalize_element, element_ptr_variant);
    });
}

template < mesh::ElementType ET,
           el_o_t            EO,
           size_t            n_rhs,
           int               system_size,
           size_t            max_dofs_per_node,
           IndexRange_c auto field_inds >
void StaticCondensationManager< CondensationPolicy::ElementBoundary >::condenseSystemImpl(
    const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >&               node_dof_map,
    tpetra_crsmatrix_t&                                                  global_mat,
    const std::array< std::span< val_t >, n_rhs >&                       global_rhs,
    const util::eigen::RowMajorSquareMatrix< lstr::val_t, system_size >& local_mat,
    const Eigen::Matrix< lstr::val_t, system_size, int{n_rhs} >&         local_rhs,
    const mesh::Element< ET, EO >&                                       element,
    util::ConstexprValue< field_inds >                                   field_inds_ctwrpr)
{
    L3STER_PROFILE_FUNCTION;
    auto&      elem_data = m_elem_data_map.at(element.id);
    const auto dof_inds  = computeLocalDofInds(element, node_dof_map, elem_data, field_inds_ctwrpr);
    const auto [row_dofs, col_dofs, rhs_dofs] = dofs::getDofsFromNodes(
        dofs::getPrimaryNodesArray< CondensationPolicy::ElementBoundary >(element), node_dof_map, field_inds_ctwrpr);

    // Primary diagonal block + RHS
    constexpr int           n_prim_dofs = std::tuple_size_v< decltype(dof_inds.primary_src_inds) >;
    thread_local const auto primary_upd_mat =
        std::make_unique< util::eigen::RowMajorSquareMatrix< val_t, n_prim_dofs > >();
    auto primary_upd_rhs = Eigen::Matrix< val_t, n_prim_dofs, int{n_rhs} >{};
    for (size_t row_ind = 0; auto src_row : dof_inds.primary_src_inds)
    {
        for (size_t col_ind = 0; auto src_col : dof_inds.primary_src_inds)
            primary_upd_mat->operator()(row_ind, col_ind++) = local_mat(src_row, src_col);
        for (size_t i = 0; i != n_rhs; ++i)
            primary_upd_rhs(row_ind, i) = local_rhs(src_row, i);
        ++row_ind;
    }
    scatterLocalSystem(*primary_upd_mat, primary_upd_rhs, global_mat, global_rhs, row_dofs, col_dofs, rhs_dofs);

    // Condensed diagonal block + RHS
    for (size_t row_ind = 0; auto src_row : dof_inds.cond_src_inds)
    {
        const auto dest_row = dof_inds.cond_dest_inds[row_ind++];
        for (size_t col_ind = 0; auto src_col : dof_inds.cond_src_inds)
        {
            const auto dest_col = dof_inds.cond_dest_inds[col_ind++];
            elem_data.diag_block(dest_row, dest_col) += local_mat(src_row, src_col);
        }
        for (size_t i = 0; i != n_rhs; ++i)
            elem_data.rhs(dest_row, i) += local_rhs(src_row, i);
    }

    // Off-diagonal block (upper)
    for (size_t row_ind = 0; auto src_row : dof_inds.primary_src_inds)
    {
        const auto dest_row = dof_inds.primary_dest_inds[row_ind++];
        for (size_t col_ind = 0; auto src_col : dof_inds.cond_src_inds)
        {
            const auto dest_col = dof_inds.cond_dest_inds[col_ind++];
            elem_data.upper_block(dest_row, dest_col) += local_mat(src_row, src_col);
        }
    }
}

template < el_o_t... orders, size_t max_dofs_per_node, size_t n_rhs >
void StaticCondensationManager< CondensationPolicy::ElementBoundary >::recoverSolutionImpl(
    const mesh::MeshPartition< orders... >&                mesh,
    const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
    const std::array< std::span< const val_t >, n_rhs >&   condensed_solutions,
    const util::ArrayOwner< size_t >&                      sol_inds,
    SolutionManager&                                       sol_man,
    const util::ArrayOwner< size_t >&                      sol_man_inds) const
{
    L3STER_PROFILE_FUNCTION;
    validateSolutionUpdateInds< max_dofs_per_node, n_rhs >(sol_inds, sol_man, sol_man_inds);
    updateSolutionPrimaryDofs(node_dof_map, condensed_solutions, sol_inds, sol_man, sol_man_inds);
    const auto dest_col_views = std::invoke([&] {
        util::ArrayOwner< std::span< val_t > > retval(sol_man_inds.size());
        std::ranges::transform(sol_man_inds, retval.begin(), [&](size_t i) { return sol_man.getFieldView(i); });
        return retval;
    });
    util::tbb::parallelFor(m_element_ids, [&](el_id_t id) {
        // Thread local variables to minimize contention on the global allocator in a parallel context
        thread_local Eigen::Matrix< val_t, Eigen::Dynamic, int{n_rhs} > primary_vals, internal_vals;

        const auto& el_data             = m_elem_data_map.at(id);
        const auto  element_ptr_variant = mesh.find(id).value();
        const auto  do_element = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >* element_ptr) {
            const auto [r_dofs, c_dofs, rhs_dofs] = dofs::getDofsFromNodes(
                dofs::getPrimaryNodesArray< CondensationPolicy::ElementBoundary >(*element_ptr), node_dof_map);
            primary_vals.resize(rhs_dofs.size(), n_rhs);
            for (size_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                for (Eigen::Index i = 0; auto dof : rhs_dofs)
                    primary_vals(i++, rhs_ind) = condensed_solutions[rhs_ind][dof];
            internal_vals = el_data.diag_block_inv * (el_data.rhs - (el_data.upper_block.transpose() * primary_vals));

            constexpr auto invalid         = std::numeric_limits< Eigen::Index >::max();
            const auto     int_dof_lookup  = std::invoke([&] {
                auto retval = std::array< Eigen::Index, max_dofs_per_node >{};
                retval.fill(invalid);
                const auto int_dof_inds = getInternalDofInds(el_data);
                for (Eigen::Index local = 0; size_t i : std::views::iota(0u, max_dofs_per_node))
                    if (std::ranges::binary_search(int_dof_inds, i))
                        retval[i] = local++;
                return retval;
            });
            const size_t   n_internal_dofs = el_data.internal_dofs_size;

            for (size_t internal_ind = 0; auto node : getInternalNodes(*element_ptr))
            {
                const auto   local_node_ind = getNodeLid(node);
                const size_t node_base      = internal_ind * n_internal_dofs;
                for (size_t dest_ind = 0; auto sol_ind : sol_inds)
                    for (size_t rhs_ind = 0; rhs_ind != n_rhs; ++rhs_ind)
                    {
                        if (int_dof_lookup[sol_ind] != invalid)
                        {
                            const auto dest_span      = dest_col_views[dest_ind];
                            const auto src_ind        = node_base + int_dof_lookup[sol_ind];
                            dest_span[local_node_ind] = internal_vals(src_ind, rhs_ind);
                        }
                        ++dest_ind;
                    }
                ++internal_ind;
            }
        };
        std::visit(do_element, element_ptr_variant);
    });
}

auto StaticCondensationManager< CondensationPolicy::ElementBoundary >::getInternalDofInds(
    const ElementCondData& cond_data) const -> std::span< const std::uint8_t >
{
    return {std::views::counted(std::next(m_internal_dof_inds.cbegin(), cond_data.internal_dofs_offs),
                                cond_data.internal_dofs_size)};
}

template < mesh::ElementType ET, el_o_t EO, size_t max_dofs_per_node, ArrayOf_c< size_t > auto field_inds >
auto StaticCondensationManager< CondensationPolicy::ElementBoundary >::computeLocalDofInds(
    const mesh::Element< ET, EO >&                         element,
    const dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
    const ElementCondData&                                 cond_data,
    util::ConstexprValue< field_inds >) -> LocalDofInds< ET, EO, field_inds.size() >
{
    constexpr auto inds_to_bmp = [](const auto& inds) {
        auto retval = std::array< bool, max_dofs_per_node >{};
        for (auto i : inds)
            retval[i] = true;
        return retval;
    };
    const auto     internal_dof_inds = getInternalDofInds(cond_data);
    constexpr auto active_inds_bmp   = inds_to_bmp(field_inds);
    const auto     internal_dof_bmp  = inds_to_bmp(internal_dof_inds);

    auto          retval            = LocalDofInds< ET, EO, field_inds.size() >{};
    size_t        primary_write_ind = 0, cond_write_ind = 0;
    std::uint32_t local_system_ind = 0, primary_ind = 0, cond_ind = 0;
    auto          boundary_ind_iter = mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds.cbegin();
    for (el_locind_t node_ind = 0; auto node : element.nodes)
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
                else if (dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >::isValid(node_dofs[i]))
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
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_STATICCONDENSATIONMANAGER_HPP
