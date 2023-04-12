#ifndef L3STER_STATICCONDENSATIONMANAGER_HPP
#define L3STER_STATICCONDENSATIONMANAGER_HPP

namespace lstr::detail
{
template < typename Derived >
class StaticCondensationManagerCRTPBase
{
public:
    void beginAssembly() { static_cast< Derived* >(this)->beginAssemblyImpl(); }
    template < size_t max_dofs_per_node >
    void endAssembly(const MeshPartition& mesh, const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map)
    {
        static_cast< Derived* >(this)->endAssemblyImpl(mesh, node_dof_map);
    }
    template < ElementTypes ET, el_o_t EO, int system_size >
    auto makeCondensedSystem(const EigenRowMajorSquareMatrix< val_t, system_size >& local_matrix,
                             const Eigen::Vector< val_t, system_size >&             local_vector,
                             const Element< ET, EO >&                               element)
    {
        return static_cast< Derived* >(this)->makeCondensedSystemImpl(local_matrix, local_vector, element);
    }
    template < size_t max_dofs_per_node, ElementTypes ET, el_o_t EO >
    auto recoverCondensedSolutionComponents(const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                                            const Element< ET, EO >&                         element,
                                            std::span< const val_t >                         condensed_solution)
    {
        return static_cast< Derived* >(this)->recoverCondensedSolutionComponentsImpl(node_dof_map);
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
    void endAssemblyImpl(const MeshPartition& mesh, const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map)
    {}
    template < ElementTypes ET, el_o_t EO, int system_size >
    auto makeCondensedSystemImpl(const EigenRowMajorSquareMatrix< val_t, system_size >& local_matrix,
                                 const Eigen::Vector< val_t, system_size >&             local_vector,
                                 const Element< ET, EO >&                               element)
        -> std::pair< const EigenRowMajorSquareMatrix< val_t, system_size >&,
                      const Eigen::Vector< val_t, system_size >& >
    {
        return std::tie(local_matrix, local_vector);
    }
    template < size_t max_dofs_per_node, ElementTypes ET, el_o_t EO >
    auto recoverCondensedSolutionComponentsImpl(const NodeToLocalDofMap< max_dofs_per_node, 3 >& node_dof_map,
                                                const Element< ET, EO >&                         element,
                                                std::span< const val_t >                         condensed_solution)
        -> std::array< std::array< val_t, max_dofs_per_node >, 0 >
    {
        return {};
    }
};

template <>
class StaticCondensationManager< CondensationPolicy::ElementBoundary > :
    public StaticCondensationManagerCRTPBase< StaticCondensationManager< CondensationPolicy::ElementBoundary > >
{};
} // namespace lstr::detail
#endif // L3STER_STATICCONDENSATIONMANAGER_HPP
