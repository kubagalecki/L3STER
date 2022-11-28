#ifndef L3STER_SOLUTIONMANAGER_HPP
#define L3STER_SOLUTIONMANAGER_HPP

#include "l3ster/assembly/NodeToDofMap.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr
{
class SolutionManager
{
    enum struct State
    {
        Owned,
        OwnedPlusShared
    };

    using map_t      = Tpetra::Map< local_dof_t, global_dof_t >;
    using importer_t = Tpetra::Import< local_dof_t, global_dof_t >;
    using fevector_t = Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >;

    static inline auto makeNodeMaps(const MeshPartition& mesh, const MpiComm& comm);
    static inline Teuchos::RCP< fevector_t >
    initNodalValues(const MeshPartition& mesh, size_t n_fields, const MpiComm& comm);

public:
    using solution_vector_t = Tpetra::Vector< val_t, local_dof_t, global_dof_t >;
    template < detail::ProblemDef_c auto problem_def >
    using node_dof_map_t = NodeToLocalDofMap< detail::deduceNFields(problem_def) >;

    inline SolutionManager(const MeshPartition& mesh, const MpiComm& comm, size_t n_fields);

    inline void communicateSharedValues();

    [[nodiscard]] inline std::span< const val_t > getNodalValues(size_t solution_ind) const;
    [[nodiscard]] inline const map_t&             getNodeMap() const;

    template < detail::ProblemDef_c auto problem_def >
    void updateSolution(const MeshPartition&                                          mesh,
                        const solution_vector_t&                                      solution,
                        const node_dof_map_t< problem_def >&                          node_dof_map,
                        std::span< const size_t, detail::deduceNFields(problem_def) > solution_inds,
                        ConstexprValue< problem_def >);

private:
    inline void toggleState();
    inline void requireState(State state, const char* except_msg) const;
    inline void prepareForMultithreadedAccessOnHost();
    inline void clearHostViews();

    Teuchos::RCP< fevector_t >                      m_nodal_values;
    State                                           m_state = State::Owned;
    Teuchos::ArrayRCP< Teuchos::ArrayRCP< val_t > > m_values_alloc;
    std::vector< std::span< val_t > >               m_values_views;          // For thread-safe access
    const map_t*                                    m_owned_plus_shared_map; // For thread-safe access
};

SolutionManager::SolutionManager(const MeshPartition& mesh, const MpiComm& comm, size_t n_fields)
    : m_nodal_values{initNodalValues(mesh, n_fields, comm)}
{
    m_nodal_values->switchActiveMultiVector();
    prepareForMultithreadedAccessOnHost();
}

inline auto SolutionManager::makeNodeMaps(const MeshPartition& mesh, const MpiComm& comm)
{
    using comm_t             = const Teuchos::MpiComm< int >;
    const auto n_owned_nodes = mesh.getNodes().size();
    const auto n_ghost_nodes = mesh.getGhostNodes().size();
    const auto n_nodes_total = n_owned_nodes + n_ghost_nodes;
    const auto nodes_alloc   = std::make_unique_for_overwrite< global_dof_t[] >(n_nodes_total);
    std::ranges::copy(mesh.getGhostNodes(), std::ranges::copy(mesh.getNodes(), nodes_alloc.get()).out);
    const auto owned_plus_shared_nodes = std::span{nodes_alloc.get(), n_nodes_total};
    const auto owned_nodes             = owned_plus_shared_nodes.subspan(0, n_owned_nodes);
    const auto make_map                = [teuchos_comm = makeTeuchosRCP< comm_t >(comm.get())](auto dofs) {
        const auto unknown_n_nodes = Teuchos::OrdinalTraits< Tpetra::global_size_t >::invalid();
        return makeTeuchosRCP< map_t >(unknown_n_nodes, asTeuchosView(dofs), 0, teuchos_comm);
    };
    return std::make_pair(make_map(owned_nodes), make_map(owned_plus_shared_nodes));
}

Teuchos::RCP< SolutionManager::fevector_t >
SolutionManager::initNodalValues(const MeshPartition& mesh, size_t n_fields, const MpiComm& comm)
{
    const auto [owned_map, owned_plus_shared_map] = makeNodeMaps(mesh, comm);
    const auto importer                           = makeTeuchosRCP< importer_t >(owned_map, owned_plus_shared_map);
    return makeTeuchosRCP< fevector_t >(owned_map, importer, n_fields);
}

template < detail::ProblemDef_c auto problem_def >
void SolutionManager::updateSolution(const MeshPartition&                                          mesh,
                                     const solution_vector_t&                                      solution,
                                     const node_dof_map_t< problem_def >&                          node_dof_map,
                                     std::span< const size_t, detail::deduceNFields(problem_def) > solution_inds,
                                     ConstexprValue< problem_def > probelm_def_ctwrapper)
{
    if (m_state == State::OwnedPlusShared)
    {
        clearHostViews();
        toggleState();
        prepareForMultithreadedAccessOnHost();
    }
    m_nodal_values->modify_host();
    const auto& dest_map        = *m_nodal_values->getMap();
    const auto  src_array_alloc = solution.getData();
    const auto  src_array       = std::span{src_array_alloc};

    const auto is_owned_node = [&](n_id_t node) {
        return not std::ranges::binary_search(mesh.getGhostNodes(), node);
    };
    const auto get_managed_span = [&](size_t problem_ind) { // Manager numbering differs from the problem's numbering
        return m_values_views[*std::next(std::ranges::begin(solution_inds), static_cast< ptrdiff_t >(problem_ind))];
    };
    const auto visit_domain = [&]< auto dom_def >(ConstexprValue< dom_def >)
    {
        constexpr auto domain                = dom_def.first;
        constexpr auto field_inds            = std::invoke([] {
            constexpr auto&          cov = dom_def.second;
            constexpr size_t         nf  = std::ranges::count(cov, true);
            std::array< size_t, nf > retval;
            std::ranges::copy_if(std::views::iota(0u, cov.size()), begin(retval), [](auto i) { return cov[i]; });
            return retval;
        });
        const auto     update_element_values = [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            const auto update_node_values = [&](n_id_t node) {
                const auto dest_local_ind = dest_map.getLocalElement(static_cast< global_dof_t >(node));
                for (size_t i = 0; auto node_dof : getValuesAtInds< field_inds >(node_dof_map(node)))
                {
                    const auto src_value = src_array[node_dof];
                    const auto dest_span = get_managed_span(field_inds[i++]);
                    std::atomic_ref{dest_span[dest_local_ind]}.store(src_value, std::memory_order_relaxed);
                }
            };
            std::ranges::for_each(element.getNodes() | std::views::filter(is_owned_node), update_node_values);
        };
        mesh.visit(update_element_values, domain, std::execution::par);
    };
    forEachConstexprParallel(visit_domain, probelm_def_ctwrapper);
}

void SolutionManager::communicateSharedValues()
{
    requireState(State::Owned, "communicateSharedValues was called in owned+shared state");
    clearHostViews();
    m_nodal_values->sync_device();
    m_nodal_values->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    toggleState();
    prepareForMultithreadedAccessOnHost();
}

std::span< const val_t > SolutionManager::getNodalValues(size_t solution_ind) const
{
    requireState(State::OwnedPlusShared, "getNodalValues was called in owned state");
    return m_values_views[solution_ind];
}

void SolutionManager::toggleState()
{
    m_nodal_values->switchActiveMultiVector();
    m_state = m_state == State::Owned ? State::OwnedPlusShared : State::Owned;
}

void SolutionManager::requireState(SolutionManager::State state, const char* except_msg) const
{
    if (m_state != state)
        throw std::runtime_error{except_msg};
}

const SolutionManager::map_t& SolutionManager::getNodeMap() const
{
    requireState(State::OwnedPlusShared, "getNodeMap was called in owned state");
    return *m_owned_plus_shared_map;
}

void SolutionManager::prepareForMultithreadedAccessOnHost()
{
    m_values_views.clear();
    m_nodal_values->sync_host();
    m_values_alloc = m_nodal_values->get2dViewNonConst();
    std::ranges::copy(m_values_alloc, std::back_inserter(m_values_views));
    m_owned_plus_shared_map = std::addressof(*m_nodal_values->getMap());
}

void SolutionManager::clearHostViews()
{
    m_values_views.clear();
    m_values_alloc = {};
}
} // namespace lstr
#endif // L3STER_SOLUTIONMANAGER_HPP
