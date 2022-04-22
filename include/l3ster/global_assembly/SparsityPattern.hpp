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
auto calculateCrsData(const MeshPartition&                                        mesh,
                      ConstexprValue< problem_def >                               problem_def_ctwrapper,
                      const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                      std::vector< global_dof_t >                                 global_dofs)
{
    thread_local std::vector< global_dof_t > scratchpad;
    constexpr auto                           merge_new_dofs = []< size_t n_new >(std::vector< global_dof_t >&             old_dofs,
                                                       const std::array< global_dof_t, n_new >& new_dofs) {
        scratchpad.resize(old_dofs.size() + n_new);
        const auto union_end = std::ranges::set_union(old_dofs, new_dofs, begin(scratchpad)).out;
        const auto union_range = std::ranges::subrange(begin(scratchpad), union_end);
        old_dofs.resize(union_range.size());
        std::ranges::copy(union_range, begin(old_dofs));
    };

    std::vector< std::vector< global_dof_t > > row_entries(global_dofs.size());
    std::vector< std::mutex >                  row_mutexes(global_dofs.size());
    const auto                                 node_to_dof_map         = GlobalNodeToDofMap{mesh, dof_intervals};
    const auto                                 global_to_local_dof_map = IndexMap{global_dofs};

    const auto process_domain = [&]< auto dom_def >(ConstexprValue< dom_def >)
    {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
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
    forConstexpr(process_domain, problem_def_ctwrapper);
    scratchpad = {};
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
