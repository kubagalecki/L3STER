#ifndef L3STER_ASSEMBLY_SPARSITYPATTERN_HPP
#define L3STER_ASSEMBLY_SPARSITYPATTERN_HPP

#include "l3ster/global_assembly/MakeTpetraMap.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"

#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"

#include "tbb/tbb.h"

namespace lstr::detail
{
template < array_of< ptrdiff_t > auto dof_inds, ElementTypes T, el_o_t O, size_t n_fields >
auto getElementDofs(const Element< T, O >& element, const GlobalNodeToDofMap< n_fields >& map)
{
    auto nodes_copy = element.getNodes();
    std::ranges::sort(nodes_copy);
    std::array< global_dof_t, dof_inds.size() * Element< T, O >::n_nodes > retval;
    for (auto insert_it = retval.begin(); auto node : nodes_copy)
    {
        const auto& full_dofs = map(node);
        for (auto ind : dof_inds)
            *insert_it++ = full_dofs[ind];
    }
    return retval;
}

template < auto problem_def >
auto calculateCrsData(const MeshPartition& mesh,
                      ConstexprValue< problem_def >,
                      const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                      std::vector< global_dof_t >                                 global_dofs)
{
    constexpr auto merge_new_dofs = [](std::vector< global_dof_t >&         old_dofs,
                                       const array_of< global_dof_t > auto& new_dofs) {
        const auto old_size = static_cast< ptrdiff_t >(old_dofs.size());
        old_dofs.resize(old_size + new_dofs.size());
        const auto old_end = std::next(begin(old_dofs), old_size);
        const auto new_end = std::set_difference(begin(new_dofs), end(new_dofs), begin(old_dofs), old_end, old_end);
        std::inplace_merge(begin(old_dofs), old_end, new_end);
        old_dofs.erase(new_end, end(old_dofs));
    };

    std::vector< std::vector< global_dof_t > > row_entries(global_dofs.size());
    std::vector< std::mutex >                  row_mutexes(global_dofs.size());
    const auto                                 node_to_dof_map         = GlobalNodeToDofMap{mesh, dof_intervals};
    const auto                                 global_to_local_dof_map = IndexMap{global_dofs};

    const auto process_domain = [&]< size_t dom_ind >(std::integral_constant< size_t, dom_ind >) {
        constexpr auto  domain_id        = problem_def[dom_ind].first;
        constexpr auto& coverage         = problem_def[dom_ind].second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto element_dofs = getElementDofs< covered_dof_inds >(element, node_to_dof_map);
            for (auto row : element_dofs)
            {
                const auto             local_row = global_to_local_dof_map(row);
                const std::scoped_lock lock{row_mutexes[local_row]};
                merge_new_dofs(row_entries[local_row], element_dofs);
            }
        };
        mesh.cvisit(process_element, {domain_id}, std::execution::par);
    };
    forConstexpr(process_domain, std::make_index_sequence< problem_def.size() >{});
    return row_entries;
}

inline Kokkos::DualView< size_t* > getRowSizes(const std::vector< std::vector< global_dof_t > >& entries)
{
    Kokkos::DualView< size_t* > retval{"sparse graph row sizes", entries.size()};
    auto                        host_view = retval.view_host();
    retval.modify_host();
    for (ptrdiff_t i = 0; const auto& entry : entries)
        host_view(i++) = entry.size();
    retval.sync_device();
    return retval;
}

template < auto problem_def >
Teuchos::RCP< const Tpetra::CrsGraph< local_dof_t, global_dof_t > >
makeSparsityPattern(const MeshPartition&                                        mesh,
                    ConstexprValue< problem_def >                               problemdef_ctwrapper,
                    const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                    const MpiComm&                                              comm)
{
    auto       owned_dofs   = detail::getNodeDofs(mesh.getNodes(), dof_intervals);
    auto       owned_map    = makeTpetraMap(owned_dofs, comm);
    const auto n_owned_dofs = owned_dofs.size();

    auto owned_plus_shared_dofs =
        concatVectors(std::move(owned_dofs), detail::getNodeDofs(mesh.getGhostNodes(), dof_intervals));
    auto owned_plus_shared_map = makeTpetraMap(owned_plus_shared_dofs, comm);
    std::ranges::inplace_merge(owned_plus_shared_dofs, std::next(begin(owned_plus_shared_dofs), n_owned_dofs));

    const auto row_entries = calculateCrsData(mesh, problemdef_ctwrapper, dof_intervals, owned_plus_shared_dofs);
    const auto row_sizes   = getRowSizes(row_entries);

    using graph_t = Tpetra::FECrsGraph< local_dof_t, global_dof_t >;
    auto retval   = Teuchos::rcp(new graph_t{owned_map, owned_plus_shared_map, row_sizes}); // NOLINT
    retval->beginAssembly();
    for (ptrdiff_t local_row = 0; local_row < static_cast< ptrdiff_t >(owned_plus_shared_dofs.size()); ++local_row)
        retval->insertGlobalIndices(owned_plus_shared_dofs[local_row], row_entries[local_row]);
    retval->endAssembly();
    return retval;
}
} // namespace lstr::detail
#endif // L3STER_ASSEMBLY_SPARSITYPATTERN_HPP