#ifndef L3STER_ASSEMBLY_MAKETPETRAMAP_HPP
#define L3STER_ASSEMBLY_MAKETPETRAMAP_HPP

#include "l3ster/global_assembly/NodeDofMaps.hpp"

#include "Tpetra_Map.hpp"

namespace lstr
{
namespace detail
{
template < size_t n_fields >
std::vector< global_dof_t > getNodeDofs(const std::vector< n_id_t >&                      sorted_nodes,
                                        const detail::node_interval_vector_t< n_fields >& dof_intervals)
{
    std::vector< global_dof_t > retval;
    retval.reserve(n_fields * sorted_nodes.size());

    const auto interval_dof_starts = detail::computeIntervalStarts(dof_intervals);

    for (auto search_it = begin(dof_intervals); auto node : sorted_nodes)
    {
        search_it                = detail::findNodeInterval(search_it, end(dof_intervals), node);
        const auto interval_ind  = std::distance(begin(dof_intervals), search_it);
        const auto& [delim, cov] = *search_it;
        const auto& [lo, hi]     = delim;
        const auto my_dof_base   = interval_dof_starts[interval_ind] + (node - lo) * cov.count();
        for (auto dof : std::views::iota(my_dof_base, my_dof_base + cov.count()))
            retval.push_back(dof);
    }

    retval.shrink_to_fit();
    return retval;
}

inline Teuchos::RCP< const Teuchos::MpiComm< int > > makeTeuchosComm(const MpiComm& comm)
{
    return Teuchos::rcp(new const Teuchos::MpiComm< int >{comm.get()}); // NOLINT
}

inline Tpetra::global_size_t getInvalidSize()
{
    return Teuchos::OrdinalTraits< Tpetra::global_size_t >::invalid();
}
} // namespace detail

template < size_t n_fields >
Teuchos::RCP< const Tpetra::Map< local_dof_t, global_dof_t > >
makeTpetraMap(const std::vector< n_id_t >&                      nodes,
              const detail::node_interval_vector_t< n_fields >& dof_intervals,
              const MpiComm&                                    comm)
{
    using map_t             = Tpetra::Map< local_dof_t, global_dof_t >;
    const auto dofs         = detail::getNodeDofs(nodes, dof_intervals);
    auto       teuchos_comm = detail::makeTeuchosComm(comm);
    return Teuchos::rcp(new const map_t{detail::getInvalidSize(), dofs, 0, teuchos_comm}); // NOLINT
}

Teuchos::RCP< const Tpetra::Map< local_dof_t, global_dof_t > > makeTpetraMap(const std::vector< global_dof_t >& dofs,
                                                                             const MpiComm&                     comm)
{
    using map_t       = Tpetra::Map< local_dof_t, global_dof_t >;
    auto teuchos_comm = detail::makeTeuchosComm(comm);
    return Teuchos::rcp(new const map_t{detail::getInvalidSize(), dofs, 0, teuchos_comm}); // NOLINT
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_MAKETPETRAMAP_HPP
