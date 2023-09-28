#ifndef L3STER_GLOB_ASM_MAKETPETRAMAP_HPP
#define L3STER_GLOB_ASM_MAKETPETRAMAP_HPP

#include "l3ster/dofs/NodeToDofMap.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr::dofs
{
template < RangeOfConvertibleTo_c< n_id_t > NodeRange, size_t n_fields, std::weakly_incrementable Iter >
auto writeNodeDofs(NodeRange&& sorted_nodes, const node_interval_vector_t< n_fields >& dof_intervals, Iter out_iter)
    -> Iter
    requires requires(global_dof_t dof) { *out_iter++ = dof; }
{
    const auto interval_dof_starts = computeIntervalStarts(dof_intervals);
    for (auto search_it = begin(dof_intervals); n_id_t node : sorted_nodes)
    {
        search_it                = findNodeInterval(search_it, end(dof_intervals), node);
        const auto interval_ind  = std::distance(begin(dof_intervals), search_it);
        const auto& [delim, cov] = *search_it;
        const auto [lo, hi]      = delim;
        const auto my_dof_base   = interval_dof_starts[interval_ind] + (node - lo) * cov.count();
        out_iter = std::ranges::copy(std::views::iota(my_dof_base, my_dof_base + cov.count()), out_iter).out;
    }
    return out_iter;
}

template < RangeOfConvertibleTo_c< n_id_t > NodeRange, size_t n_fields >
auto getNodeDofs(NodeRange&& sorted_nodes, const node_interval_vector_t< n_fields >& dof_intervals)
    -> std::vector< global_dof_t >
{
    auto retval = std::vector< global_dof_t >{};
    if constexpr (std::ranges::sized_range< NodeRange >)
        retval.reserve(std::ranges::size(sorted_nodes) * n_fields);
    writeNodeDofs(std::forward< NodeRange >(sorted_nodes), dof_intervals, std::back_inserter(retval));
    retval.shrink_to_fit();
    return retval;
}

inline auto makeTeuchosMpiComm(const MpiComm& comm) -> Teuchos::RCP< const Teuchos::MpiComm< int > >
{
    return util::makeTeuchosRCP< const Teuchos::MpiComm< int > >(comm.get());
}

inline auto getInvalidSize() -> Tpetra::global_size_t
{
    return Teuchos::OrdinalTraits< Tpetra::global_size_t >::invalid();
}

inline auto makeTpetraMap(std::span< const global_dof_t > dofs, const MpiComm& comm)
    -> Teuchos::RCP< const tpetra_map_t >
{
    auto       teuchos_comm      = makeTeuchosMpiComm(comm);
    const auto compute_size      = getInvalidSize();
    const auto dofs_teuchos_view = util::asTeuchosView(dofs);
    return util::makeTeuchosRCP< const tpetra_map_t >(compute_size, dofs_teuchos_view, 0, std::move(teuchos_comm));
}

template < RangeOfConvertibleTo_c< n_id_t > Nodes, size_t n_fields >
auto makeTpetraMap(Nodes&& nodes, const node_interval_vector_t< n_fields >& dof_intervals, const MpiComm& comm)
    -> Teuchos::RCP< const tpetra_map_t >
{
    const auto dofs = getNodeDofs(std::forward< Nodes >(nodes), dof_intervals);
    return makeTpetraMap(dofs, comm);
}
} // namespace lstr::dofs
#endif // L3STER_GLOB_ASM_MAKETPETRAMAP_HPP
