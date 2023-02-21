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

    static inline auto makeNodeMaps(const MeshPartition& mesh, const MpiComm& comm);
    static inline Teuchos::RCP< tpetra_femultivector_t >
    initNodalValues(const MeshPartition& mesh, size_t n_fields, const MpiComm& comm);

public:
    template < detail::ProblemDef_c auto problem_def >
    using node_dof_map_t = NodeToLocalDofMap< detail::deduceNFields(problem_def), 3 >;

    inline SolutionManager(const MeshPartition& mesh, const MpiComm& comm, size_t n_fields);

    template < size_t N >
    [[nodiscard]] auto getNodeValues(n_id_t node, std::span< const size_t, N > field_inds) const
        requires(N != std::dynamic_extent);
    [[nodiscard]] inline std::span< const val_t > getFieldView(size_t field_ind) const;
    [[nodiscard]] inline const tpetra_map_t&      getNodeMap() const;

    template < detail::ProblemDef_c auto problem_def >
    void        updateSolution(const MeshPartition&                                          mesh,
                               const tpetra_vector_t&                                        solution,
                               const node_dof_map_t< problem_def >&                          node_dof_map,
                               std::span< const size_t, detail::deduceNFields(problem_def) > solution_inds,
                               ConstexprValue< problem_def >);
    inline void communicateSharedValues();

private:
    inline void toggleState();
    inline void assertState(State expected, const char* err_msg) const;

    State                                          m_state{State::Owned};
    Teuchos::RCP< tpetra_femultivector_t >         m_nodal_values;
    tpetra_femultivector_t::dual_view_type::t_host m_nodal_values_view;     // For thread-safe access
    const tpetra_map_t*                            m_owned_plus_shared_map; // For thread-safe access
};

SolutionManager::SolutionManager(const MeshPartition& mesh, const MpiComm& comm, size_t n_fields)
    : m_nodal_values{initNodalValues(mesh, n_fields, comm)}, m_owned_plus_shared_map{m_nodal_values->getMap().get()}
{
    m_nodal_values->switchActiveMultiVector();
    m_nodal_values_view = m_nodal_values->getLocalViewHost(Tpetra::Access::ReadWrite);
}

inline auto SolutionManager::makeNodeMaps(const MeshPartition& mesh, const MpiComm& comm)
{
    using comm_t             = const Teuchos::MpiComm< int >;
    const auto n_owned_nodes = mesh.getOwnedNodes().size();
    const auto n_ghost_nodes = mesh.getGhostNodes().size();
    const auto n_nodes_total = n_owned_nodes + n_ghost_nodes;
    auto       nodes         = ArrayOwner< global_dof_t >(n_nodes_total);
    std::ranges::copy(mesh.getAllNodes(), nodes.begin()); // n_id_t != global_dof_t
    const auto all_nodes   = std::span{nodes.begin(), n_nodes_total};
    const auto owned_nodes = all_nodes.subspan(0, n_owned_nodes);
    const auto make_map    = [teuchos_comm = makeTeuchosRCP< comm_t >(comm.get())](auto dofs) {
        const auto unknown_n_nodes = Teuchos::OrdinalTraits< Tpetra::global_size_t >::invalid();
        return makeTeuchosRCP< const tpetra_map_t >(unknown_n_nodes, asTeuchosView(dofs), 0, teuchos_comm);
    };
    return std::make_pair(make_map(owned_nodes), make_map(all_nodes));
}

Teuchos::RCP< tpetra_femultivector_t >
SolutionManager::initNodalValues(const MeshPartition& mesh, size_t n_fields, const MpiComm& comm)
{
    auto [owned_map, all_map] = makeNodeMaps(mesh, comm);
    auto importer             = makeTeuchosRCP< tpetra_import_t >(owned_map, std::move(all_map));
    return makeTeuchosRCP< tpetra_femultivector_t >(std::move(owned_map), std::move(importer), n_fields);
}

template < detail::ProblemDef_c auto problem_def >
void SolutionManager::updateSolution(const MeshPartition&                                          mesh,
                                     const tpetra_vector_t&                                        solution,
                                     const node_dof_map_t< problem_def >&                          node_dof_map,
                                     std::span< const size_t, detail::deduceNFields(problem_def) > solution_inds,
                                     ConstexprValue< problem_def > probelm_def_ctwrapper)
{
    if (m_state == State::OwnedPlusShared)
    {
        toggleState();
        m_nodal_values_view = m_nodal_values->getLocalViewHost(Tpetra::Access::ReadWrite);
    }
    const auto solution_view = Kokkos::subview(solution.getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    const auto visit_domain  = [&]< auto dom_def >(ConstexprValue< dom_def >) {
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
                const auto dest_row = m_owned_plus_shared_map->getLocalElement(static_cast< global_dof_t >(node));
                for (size_t i = 0; auto node_dof : getValuesAtInds< field_inds >(node_dof_map(node).front()))
                {
                    const auto src_value = solution_view[node_dof];
                    const auto dest_col  = solution_inds[field_inds[i++]];
                    auto&      dest      = m_nodal_values_view(dest_row, dest_col);
                    std::atomic_ref{dest}.store(src_value, std::memory_order_relaxed);
                }
            };
            std::ranges::for_each(element.getNodes() |
                                      std::views::filter([&](n_id_t node) { return not mesh.isGhostNode(node); }),
                                  update_node_values);
        };
        mesh.visit(update_element_values, domain, std::execution::par);
    };
    forEachConstexprParallel(visit_domain, probelm_def_ctwrapper);
}

void SolutionManager::communicateSharedValues()
{
    assertState(State::Owned, "SolutionManager::communicateSharedValues was called in owned+shared state");
    m_nodal_values->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    toggleState();
    m_nodal_values_view = m_nodal_values->getLocalViewHost(Tpetra::Access::ReadWrite);
}

template < size_t N >
[[nodiscard]] auto SolutionManager::getNodeValues(n_id_t node, std::span< const size_t, N > field_inds) const
    requires(N != std::dynamic_extent)
{
    assertState(State::OwnedPlusShared, "SolutionManager::getNodeValues was called in owned state");
    const auto             local_ind = m_owned_plus_shared_map->getLocalElement(static_cast< global_dof_t >(node));
    std::array< val_t, N > retval;
    std::ranges::transform(field_inds, begin(retval), [&](size_t i) { return m_nodal_values_view(local_ind, i); });
    return retval;
}

std::span< const val_t > SolutionManager::getFieldView(size_t field_ind) const
{
    assertState(State::OwnedPlusShared, "SolutionManager::getFieldView was called in owned state");
    return asSpan(Kokkos::subview(m_nodal_values_view, Kokkos::ALL, field_ind));
}

void SolutionManager::toggleState()
{
    m_nodal_values->switchActiveMultiVector();
    m_state = m_state == State::Owned ? State::OwnedPlusShared : State::Owned;
}

void SolutionManager::assertState(State expected, const char* err_msg) const
{
    if (m_state != expected)
        throw std::runtime_error{err_msg};
}

const tpetra_map_t& SolutionManager::getNodeMap() const
{
    assertState(State::OwnedPlusShared, "SolutionManager::getNodeMap was called in owned state");
    return *m_owned_plus_shared_map;
}
} // namespace lstr
#endif // L3STER_SOLUTIONMANAGER_HPP
