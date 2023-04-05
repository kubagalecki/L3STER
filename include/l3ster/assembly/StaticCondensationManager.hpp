#ifndef L3STER_STATICCONDENSATIONMANAGER_HPP
#define L3STER_STATICCONDENSATIONMANAGER_HPP

namespace lstr::detail
{
template < typename Derived >
class StaticCondensationManagerCRTPBase
{
public:
    template < size_t max_dofs_per_node, ElementTypes ET, el_o_t EO >
    auto recoverUncondensedSolution(const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                                    const Element< ET, EO >&                         element,
                                    std::span< const val_t >                         condensed_solution)
    {
        return static_cast< Derived* >(this)->recoverUncondensedSolutionImpl(node_dof_map);
    }
};

template < CondensationPolicy CP >
class StaticCondensationManager;

template <>
class StaticCondensationManager< CondensationPolicy::None > :
    public StaticCondensationManagerCRTPBase< StaticCondensationManager< CondensationPolicy::None > >
{};

template <>
class StaticCondensationManager< CondensationPolicy::ElementBoundary > :
    public StaticCondensationManagerCRTPBase< StaticCondensationManager< CondensationPolicy::ElementBoundary > >
{};
} // namespace lstr::detail
#endif // L3STER_STATICCONDENSATIONMANAGER_HPP
