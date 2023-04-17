#ifndef L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP
#define L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP

#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/util/TbbUtils.hpp"

#include <shared_mutex>

namespace lstr::detail
{
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

template < CondensationPolicy CP >
class StaticCondensationManager;

template <>
class StaticCondensationManager< CondensationPolicy::None > :
    public StaticCondensationManagerInterface< StaticCondensationManager< CondensationPolicy::None > >
{
public:
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
    template < size_t max_dofs_per_node, ProblemDef_c auto problem_def >
    StaticCondensationManager(const MeshPartition&                             mesh,
                              const NodeToLocalDofMap< max_dofs_per_node, 3 >& dof_map,
                              ConstexprValue< problem_def >);

    void beginAssemblyImpl()
    {
        for (auto& [id, payload] : m_elem_data_map)
        {
            payload.diag_block.setZero();
            payload.upper_block.setZero();
        }
    }
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
    {}
    template < size_t max_dofs_per_node >
    void recoverSolutionImpl(const MeshPartition&,
                             const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                             std::span< const val_t >                         condensed_solution,
                             IndexRange_c auto&&                              sol_inds,
                             SolutionManager&                                 sol_man,
                             IndexRange_c auto&&                              sol_man_inds) const
    {}

private:
    struct ElementCondData
    {
        std::span< const std::uint8_t >                         internal_dof_inds;
        eigen::DynamicallySizedMatrix< val_t, Eigen::RowMajor > diag_block, upper_block;
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
    mesh.visit(
        [&dof_map, this]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            using dof_bmp_t    = std::bitset< max_dofs_per_node >;
            auto dof_bmp_range = getBoundaryNodes(element) | std::views::transform(dof_map) | std::views::keys |
                                 std::views::transform([&](const std::array< local_dof_t, max_dofs_per_node >& dofs) {
                                     auto retval = dof_bmp_t{};
                                     for (size_t i = 0; auto dof : dofs)
                                     {
                                         retval[i] = dof != NodeToLocalDofMap< max_dofs_per_node, 3 >::invalid_dof;
                                         ++i;
                                     }
                                     return retval;
                                 }) |
                                 std::views::common;
            size_t n_boundary_dofs{};
            auto   internal_node_dof_bmp = dof_bmp_t{};
            for (size_t i = 0; i != internal_node_dof_bmp.size(); ++i)
                internal_node_dof_bmp[i] = true;
            for (auto dof_bmp : dof_bmp_range)
            {
                internal_node_dof_bmp &= dof_bmp;
                n_boundary_dofs += dof_bmp.count();
            }
            auto el_int_dofs_begin = m_internal_dof_inds.end();
            for (size_t i = 0; i != internal_node_dof_bmp.size(); ++i)
                if (internal_node_dof_bmp[i])
                    m_internal_dof_inds.push_back(static_cast< std::uint8_t >(i));
            auto       el_int_dofs_end = m_internal_dof_inds.end();
            const auto n_internal_dofs = (ElementTraits< Element< ET, EO > >::nodes_per_element -
                                          ElementTraits< Element< ET, EO > >::boundary_node_inds.size()) *
                                         internal_node_dof_bmp.count();
            m_elem_data_map.emplace(element.getId(),
                                    ElementCondData{.internal_dof_inds{el_int_dofs_begin, el_int_dofs_end},
                                                    .diag_block{n_internal_dofs, n_internal_dofs},
                                                    .upper_block{n_internal_dofs, n_internal_dofs + n_boundary_dofs}});
        },
        problem_def | std::views::keys,
        std::execution::seq);
    m_internal_dof_inds.shrink_to_fit();
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP
