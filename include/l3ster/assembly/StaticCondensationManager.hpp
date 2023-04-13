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
                const auto dof = dofs[sol_ind];
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
class StaticCondensationManagerCRTPBase
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
    void condenseSystem(const NodeToLocalDofMap< max_dofs_per_node, 3 >&       node_dof_map,
                        tpetra_crsmatrix_t&                                    global_mat,
                        std::span< val_t >                                     global_rhs,
                        const EigenRowMajorSquareMatrix< val_t, system_size >& local_matrix,
                        const Eigen::Vector< val_t, system_size >&             local_vector,
                        const Element< ET, EO >&                               element,
                        ConstexprValue< field_inds >                           field_inds_ctwrpr)
    {
        static_cast< Derived* >(this)->condenseSystemImpl(
            node_dof_map, global_mat, global_rhs, local_matrix, local_vector, element, field_inds_ctwrpr);
    }
    template < size_t max_dofs_per_node, ElementTypes ET, el_o_t EO >
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
    public StaticCondensationManagerCRTPBase< StaticCondensationManager< CondensationPolicy::None > >
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
    void condenseSystemImpl(const NodeToLocalDofMap< max_dofs_per_node, 3 >&       node_dof_map,
                            tpetra_crsmatrix_t&                                    global_mat,
                            std::span< val_t >                                     global_rhs,
                            const EigenRowMajorSquareMatrix< val_t, system_size >& local_mat,
                            const Eigen::Vector< val_t, system_size >&             local_vec,
                            const Element< ET, EO >&                               element,
                            ConstexprValue< field_inds >                           field_inds_ctwrpr)
    {
        const auto [row_dofs, col_dofs, rhs_dofs] =
            detail::getUnsortedPrimaryDofs(element, node_dof_map, no_condensation, field_inds_ctwrpr);
        detail::scatterLocalSystem(local_mat, local_vec, global_mat, global_rhs, row_dofs, col_dofs, rhs_dofs);
    }
    template < size_t max_dofs_per_node, ElementTypes ET, el_o_t EO >
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
    public StaticCondensationManagerCRTPBase< StaticCondensationManager< CondensationPolicy::ElementBoundary > >
{};
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_STATICCONDENSATIONMANAGER_HPP
