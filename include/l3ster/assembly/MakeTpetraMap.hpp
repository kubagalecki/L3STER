#ifndef L3STER_ASSEMBLY_MAKETPETRAMAP_HPP
#define L3STER_ASSEMBLY_MAKETPETRAMAP_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr
{
namespace detail
{
template < size_t n_fields >
auto writeNodeDofs(RangeOfConvertibleTo_c< n_id_t > auto&&   sorted_nodes,
                   const node_interval_vector_t< n_fields >& dof_intervals,
                   std::weakly_incrementable auto            out_iter) -> decltype(out_iter)
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

template < size_t n_fields >
auto getNodeDofs(RangeOfConvertibleTo_c< n_id_t > auto&&   sorted_nodes,
                 const node_interval_vector_t< n_fields >& dof_intervals) -> std::vector< global_dof_t >
{
    std::vector< global_dof_t > retval;
    if constexpr (std::ranges::sized_range< decltype(sorted_nodes) >)
        retval.reserve(std::ranges::size(sorted_nodes) * n_fields);
    writeNodeDofs(std::forward< decltype(sorted_nodes) >(sorted_nodes), dof_intervals, std::back_inserter(retval));
    retval.shrink_to_fit();
    return retval;
}

inline Teuchos::RCP< const Teuchos::MpiComm< int > > makeTeuchosMpiComm(const MpiComm& comm)
{
    return util::makeTeuchosRCP< const Teuchos::MpiComm< int > >(comm.get());
}

inline Tpetra::global_size_t getInvalidSize()
{
    return Teuchos::OrdinalTraits< Tpetra::global_size_t >::invalid();
}
} // namespace detail

inline Teuchos::RCP< const tpetra_map_t > makeTpetraMap(std::span< const global_dof_t > dofs, const MpiComm& comm)
{
    auto       teuchos_comm      = detail::makeTeuchosMpiComm(comm);
    const auto compute_size      = detail::getInvalidSize();
    const auto dofs_teuchos_view = util::asTeuchosView(dofs);
    return util::makeTeuchosRCP< const tpetra_map_t >(compute_size, dofs_teuchos_view, 0, std::move(teuchos_comm));
}

template < size_t n_fields >
Teuchos::RCP< const tpetra_map_t > makeTpetraMap(RangeOfConvertibleTo_c< n_id_t > auto&&           nodes,
                                                 const detail::node_interval_vector_t< n_fields >& dof_intervals,
                                                 const MpiComm&                                    comm)
{
    const auto dofs = detail::getNodeDofs(std::forward< decltype(nodes) >(nodes), dof_intervals);
    return makeTpetraMap(dofs, comm);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_MAKETPETRAMAP_HPP
