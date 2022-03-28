#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEM_HPP

#include "l3ster/global_assembly/MakeTpetraMap.hpp"
#include "l3ster/util/DynamicBitset.hpp"

#include "Tpetra_FECrsMatrix.hpp"
#include "Tpetra_FEMultiVector.hpp"

namespace lstr
{
namespace detail
{
size_t getChunkSize(size_t n_dofs)
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

template < size_t                                                                n_fields,
           size_t                                                                n_domains,
           std::array< Pair< d_id_t, std::array< bool, n_fields > >, n_domains > problem_def >
Kokkos::DualView< size_t* > calculateRowSizes(const MeshPartition&                      mesh,
                                              ConstexprValue< problem_def >             problemdef_ctwrapper,
                                              const node_interval_vector_t< n_fields >& dof_intervals,
                                              size_t                                    n_dofs)
{
    Kokkos::DualView< size_t* > retval{"row_sizes", n_dofs};
    retval.modify_host();
    auto host_view = retval.view_host();

    const auto dof_map = GlobalNodeToDofMap< n_fields >{mesh, dof_intervals};

    const auto    chunk_size = getChunkSize(n_dofs);
    DynamicBitset nonzero_inds{chunk_size * n_dofs};
    auto          nonzero_inds_atomic = nonzero_inds.getAtomicView();

    for (size_t chunk_begin = 0; chunk_begin < n_dofs; chunk_begin += chunk_size)
    {
        const auto chunk_end          = std::min(chunk_begin + chunk_size, n_dofs);
        const auto current_chunk_size = chunk_end - chunk_begin;
        nonzero_inds.resize(current_chunk_size * n_dofs);
        nonzero_inds.clear();

        forConstexpr(
            [&]< auto I >(std::integral_constant< decltype(I), I >) {
                constexpr auto  domain           = problem_def[I].first;
                constexpr auto& coverage         = problem_def[I].second;
                constexpr auto  covered_dof_inds = getTrueInds< coverage >();

                mesh.cvisit(
                    [&]< ElementTypes T, el_o_t O >(const Element< T, O >& el) {
                        const auto el_dofs = getElementDofs< covered_dof_inds >(el, dof_map);
                        for (size_t chunk_row = 0; chunk_row < current_chunk_size; ++chunk_row)
                        {
                            const size_t row_inds_begin = chunk_row * n_dofs;
                            for (auto dof : el_dofs)
                                nonzero_inds_atomic.set(row_inds_begin + dof, std::memory_order_relaxed);
                        }
                    },
                    {domain},
                    std::execution::par);
            },
            std::make_index_sequence< problem_def.size() >{});

        for (size_t chunk_row = 0; chunk_row < current_chunk_size; ++chunk_row)
        {
            auto row_inds                      = nonzero_inds.getSubView(chunk_row, chunk_row + n_dofs);
            host_view(chunk_begin + chunk_row) = row_inds.count();
        }
    }
    retval.sync_device();
    return retval;
}

template < size_t                                                                n_fields,
           size_t                                                                n_domains,
           std::array< Pair< d_id_t, std::array< bool, n_fields > >, n_domains > problem_def >
Teuchos::RCP< const Tpetra::CrsGraph< local_dof_t, global_dof_t > >
makeSparsityPattern(const MeshPartition&                      mesh,
                    ConstexprValue< problem_def >             problemdef_ctwrapper,
                    const node_interval_vector_t< n_fields >& dof_intervals,
                    const MpiComm&                            comm)
{
    const auto owned_dofs             = detail::getNodeDofs(mesh.getNodes(), dof_intervals);
    const auto owned_plus_shared_dofs = [&] {
        const auto shared_dofs = detail::getNodeDofs(mesh.getGhostNodes(), dof_intervals);
        concatVectors(owned_dofs, shared_dofs);
    }();

    auto owned_map             = makeTpetraMap(owned_dofs, comm);
    auto owned_plus_shared_map = makeTpetraMap(owned_plus_shared_dofs, comm);

    const auto row_sizes =
        calculateRowSizes(mesh, problemdef_ctwrapper, dof_intervals, owned_plus_shared_dofs->getNodeNumElements());
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
