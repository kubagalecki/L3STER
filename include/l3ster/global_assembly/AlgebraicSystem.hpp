#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP

#include "l3ster/global_assembly/MakeTpetraMap.hpp"
#include "l3ster/util/DynamicBitset.hpp"
#include "l3ster/util/IndexMap.hpp"

#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"

#include "tbb/tbb.h"

namespace lstr
{
namespace detail
{
inline size_t getChunkSize(size_t n_dofs)
{
    constexpr size_t max_memory = 1ul << 32;
    return std::clamp(max_memory / n_dofs, size_t{1}, n_dofs);
}

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
    constexpr auto init_row_entries = [](size_t n) {
        constexpr float                            assumed_fill_factor = .001;
        const size_t                               assumed_row_entries = std::llround(assumed_fill_factor * n);
        std::vector< std::vector< global_dof_t > > row_entries;
        for (auto& vec : row_entries)
            vec.reserve(assumed_row_entries);
        return row_entries;
    };
    auto row_entries = init_row_entries(global_dofs.size());

    constexpr auto merge_new_dofs = [](std::vector< global_dof_t >&         dofs,
                                       const array_of< global_dof_t > auto& new_dofs) {
        // Both old and new dofs are assumed to be sorted and unique;
        const auto old_size = static_cast< ptrdiff_t >(dofs.size());
        dofs.resize(old_size + new_dofs.size());
        const auto old_end = std::next(begin(dofs), old_size);
        const auto new_end = std::set_difference(begin(new_dofs), end(new_dofs), begin(dofs), old_end, old_end);
        std::inplace_merge(begin(dofs), old_end, new_end);
        dofs.erase(new_end, end(dofs));
    };

    const auto node_to_dof_map         = GlobalNodeToDofMap{mesh, dof_intervals};
    const auto global_to_local_dof_map = IndexMap{global_dofs};

    const auto add_row_entries = [&](const tbb::blocked_range< global_dof_t >& row_range) {
        const auto process_domain = [&]< size_t dom_ind >(std::integral_constant< decltype(dom_ind), dom_ind >) {
            constexpr auto  domain_id        = problem_def[dom_ind].first;
            constexpr auto& coverage         = problem_def[dom_ind].second;
            constexpr auto  covered_dof_inds = getTrueInds< coverage >();

            const global_dof_t row_begin = row_range.begin(), row_end = row_range.end();

            const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                const auto element_dofs = getElementDofs< covered_dof_inds >(element, node_to_dof_map);
                std::array< global_dof_t, std::tuple_size_v< decltype(element_dofs) > > inrange_dofs;
                const auto [loc_row_begin, loc_row_end] =
                    std::ranges::copy_if(element_dofs, begin(inrange_dofs), [&](global_dof_t dof_global) {
                        const auto dof_local = global_to_local_dof_map(dof_global);
                        return dof_local >= row_begin and dof_local < row_end;
                    });

                for (auto row : std::ranges::subrange(loc_row_begin, loc_row_end))
                    merge_new_dofs(row_entries[row], element_dofs);
            };
            mesh.cvisit(process_element, {domain_id}, std::execution::par);
        };
        forConstexpr(process_domain, std::make_index_sequence< problem_def.size() >{});
    };
    const auto row_range = tbb::blocked_range< global_dof_t >{0, static_cast< global_dof_t >(global_dofs.size())};
    tbb::parallel_for(row_range, add_row_entries);

    return row_entries;
}

/*
template < auto problem_def >
auto calculateCrsData(const MeshPartition& mesh,
                      ConstexprValue< problem_def >,
                      const node_interval_vector_t< deduceNFields(problem_def) >& dof_intervals,
                      size_t                                                      n_dofs)
{
    Kokkos::DualView< size_t* > row_sizes{"row_sizes", n_dofs};
    row_sizes.modify_host();
    auto                        host_view = row_sizes.view_host();
    std::vector< global_dof_t > entries;

    const auto dof_map = GlobalNodeToDofMap< deduceNFields(problem_def) >{mesh, dof_intervals};

    const auto    chunk_size = getChunkSize(n_dofs);
    DynamicBitset nonzero_inds{chunk_size * n_dofs};
    auto          nonzero_inds_atomic = nonzero_inds.getAtomicView();

    const auto process_chunk = [&](size_t chunk_begin, size_t chunk_end) {
        const size_t current_chunk_size = chunk_end - chunk_begin;
        nonzero_inds.resize(current_chunk_size * n_dofs);
        nonzero_inds.clear();

        const auto process_domain = [&]< size_t dom_ind >(std::integral_constant< decltype(dom_ind), dom_ind >) {
            constexpr auto  domain_id        = problem_def[dom_ind].first;
            constexpr auto& coverage         = problem_def[dom_ind].second;
            constexpr auto  covered_dof_inds = getTrueInds< coverage >();

            const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                const auto element_dofs = getElementDofs< covered_dof_inds >(element, dof_map);
                for (size_t chunk_row = 0; chunk_row < current_chunk_size; ++chunk_row)
                {
                    const size_t row_inds_begin = chunk_row * n_dofs;
                    for (auto dof : element_dofs)
                        nonzero_inds_atomic.set(row_inds_begin + dof, std::memory_order_relaxed);
                }
            };
            mesh.cvisit(process_element, {domain_id}, std::execution::par);
        };
        forConstexpr(process_domain, std::make_index_sequence< problem_def.size() >{});

        auto chunk_row_sizes =
            Kokkos::subview(host_view, std::make_pair(chunk_begin, chunk_begin + current_chunk_size));
        tbb::parallel_for(tbb::blocked_range< size_t >{0, current_chunk_size},
                          [&](const tbb::blocked_range< size_t >& range) {
                              for (size_t chunk_row = range.begin(); chunk_row != range.end(); ++chunk_row)
                              {
                                  const size_t row_begin = chunk_row * n_dofs, row_end = (chunk_row + 1) * n_dofs;
                                  const auto   row_inds      = nonzero_inds.getSubView(row_begin, row_end);
                                  chunk_row_sizes(chunk_row) = row_inds.count();
                              }
                          });

        Kokkos::View< size_t*, Kokkos::HostSpace > chunk_row_start_inds{"chunk begin inds", chunk_row_sizes.size()};
        Kokkos::parallel_scan(
            "Chunk row entry offset calculation",
            Kokkos::RangePolicy< Kokkos::DefaultHostExecutionSpace >{size_t{0}, current_chunk_size},
            KOKKOS_LAMBDA(size_t i, size_t & lsum, bool final) {
                if (final)
                    chunk_row_start_inds(i) = lsum;
                lsum += chunk_row_sizes(i);
            });
        const size_t n_chunk_entries = chunk_row_start_inds(chunk_size - 1) + chunk_row_sizes(chunk_size - 1);

        if (chunk_begin == 0)
            entries.reserve(n_chunk_entries * (n_dofs / current_chunk_size + 1));

        const auto n_previous_entries = entries.size();
        entries.resize(n_previous_entries + n_chunk_entries);
        std::span chunk_entries{std::next(entries.begin(), static_cast< ptrdiff_t >(n_previous_entries)),
                                n_chunk_entries};

        tbb::parallel_for(
            tbb::blocked_range< size_t >{0, current_chunk_size}, [&](const tbb::blocked_range< size_t >& range) {
                size_t n_range_entries = 0;
                for (size_t chunk_row = range.begin(); chunk_row != range.end(); ++chunk_row)
                    n_range_entries += chunk_row_sizes(chunk_row);
                std::vector< global_dof_t > range_entries(n_range_entries);
                for (size_t chunk_row = range.begin(); chunk_row != range.end(); ++chunk_row)
                {
                    const size_t row_begin = chunk_row * n_dofs, row_end = (chunk_row + 1) * n_dofs;
                    const auto   row_inds = nonzero_inds.getSubView(row_begin, row_end);
                    auto         insertion_it =
                        std::next(chunk_entries.begin(), static_cast< ptrdiff_t >(chunk_row_start_inds(chunk_row)));
                    for (size_t i = 0; i < n_dofs; ++i)
                        if (row_inds.test(i))
                            *insertion_it++ = static_cast< global_dof_t >(i);
                }
            });
    };
    for (size_t chunk_begin = 0; chunk_begin < n_dofs; chunk_begin += chunk_size)
        process_chunk(chunk_begin, std::min(chunk_begin + chunk_size, n_dofs));

    row_sizes.sync_device();
    return std::make_pair(row_sizes, std::move(entries));
}
*/

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

    const auto [row_sizes, dofs] = calculateCrsData(mesh, problemdef_ctwrapper, dof_intervals, owned_plus_shared_dofs);

    return Teuchos::rcp(
        new const Tpetra::CrsGraph< local_dof_t, global_dof_t >{owned_map, owned_plus_shared_map, row_sizes}); // NOLINT
}
} // namespace detail

class AlgebraicSystem
{
public:
    using matrix_t = Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t >;
    using vector_t = Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >;

    template < size_t n_fields >
    AlgebraicSystem(const MeshPartition&                              partition,
                    const detail::node_interval_vector_t< n_fields >& dof_intervals,
                    const MpiComm&                                    comm);

private:
    Teuchos::RCP< matrix_t > matrix;
    Teuchos::RCP< vector_t > vector;
};

template < size_t n_fields >
AlgebraicSystem::AlgebraicSystem(const MeshPartition&                              partition,
                                 const detail::node_interval_vector_t< n_fields >& dof_intervals,
                                 const MpiComm&                                    comm)
{}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP
