#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP

#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"

namespace lstr
{
template < size_t n_fields >
class AlgebraicSystemManager
{
    using map_t          = Tpetra::Map< local_dof_t, global_dof_t >;
    using fegraph_t      = Tpetra::FECrsGraph< local_dof_t, global_dof_t >;
    using fematrix_t     = Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t >;
    using fevector_t     = Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >;
    using mltvector_t    = Tpetra::MultiVector< val_t, local_dof_t, global_dof_t >;
    using vector_t       = Tpetra::Vector< val_t, local_dof_t, global_dof_t >;
    using dof_map_local  = NodeToLocalDofMap< n_fields >;
    using dof_map_global = NodeToGlobalDofMap< n_fields >;

public:
    template < detail::ProblemDef_c auto problem_def >
    AlgebraicSystemManager(const MpiComm& comm, const MeshPartition& mesh, ConstexprValue< problem_def >);
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    AlgebraicSystemManager(const MpiComm&       comm,
                           const MeshPartition& mesh,
                           ConstexprValue< problem_def >,
                           ConstexprValue< dirichlet_def >);

    const auto& getMatrix() const { return m_matrix; }
    const auto& getRhs() const { return m_rhs; }
    const auto& getRowMap() const { return m_node_to_row_dof_map; }
    const auto& getColMap() const { return m_node_to_col_dof_map; }
    const auto& getRhsMap() const { return m_node_to_rhs_dof_map; }

    [[nodiscard]] inline Teuchos::RCP< mltvector_t > makeSolutionMultiVector(size_t n_cols = 1) const;

    inline void beginAssembly();
    inline void endAssembly();
    inline void beginModify();
    inline void endModify();
    inline void setToZero();

    template < BasisTypes              BT,
               QuadratureTypes         QT,
               q_o_t                   QO,
               ArrayOf_c< size_t > auto field_inds,
               typename Kernel,
               detail::FieldValGetter_c FvalGetter,
               detail::DomainIdRange_c  R >
    void assembleDomainProblem(
        Kernel&& kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& fval_getter, val_t time = 0.);
    template < BasisTypes              BT,
               QuadratureTypes         QT,
               q_o_t                   QO,
               ArrayOf_c< size_t > auto field_inds,
               typename Kernel,
               detail::FieldValGetter_c FvalGetter >
    void
    assembleBoundaryProblem(Kernel&& kernel, const BoundaryView& boundary, FvalGetter&& fval_getter, val_t time = 0.);

    inline void applyDirichletBCs(const vector_t& bc_vals);

private:
    template < detail::ProblemDef_c auto problem_def >
    dof_map_global initSystem(const MpiComm& comm, const MeshPartition& mesh, ConstexprValue< problem_def >);
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    void        initDirichletBCs(const MeshPartition&                  mesh,
                                 const NodeToGlobalDofMap< n_fields >& global_node_dof_map,
                                 ConstexprValue< problem_def >,
                                 ConstexprValue< dirichlet_def >);
    inline void openRhs();
    inline void closeRhs();

    enum class State
    {
        OpenForAssembly,
        OpenForModify,
        Closed
    };

    dof_map_local                         m_node_to_row_dof_map, m_node_to_col_dof_map, m_node_to_rhs_dof_map;
    Teuchos::RCP< fematrix_t >            m_matrix;
    Teuchos::RCP< fevector_t >            m_rhs;
    std::span< val_t >                    m_rhs_view; // We need a thread safe view, which Teuchos::ArrayRCP is not
    Teuchos::RCP< const fegraph_t >       m_sparsity_graph;
    std::optional< DirichletBCAlgebraic > m_dirichlet_bcs;
    State                                 m_state;
};

template < detail::ProblemDef_c auto problem_def >
AlgebraicSystemManager(const MpiComm&, const MeshPartition&, ConstexprValue< problem_def >)
    -> AlgebraicSystemManager< detail::deduceNFields(problem_def) >;
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
AlgebraicSystemManager(const MpiComm&,
                       const MeshPartition&,
                       ConstexprValue< problem_def >,
                       ConstexprValue< dirichlet_def >) -> AlgebraicSystemManager< detail::deduceNFields(problem_def) >;

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def >
AlgebraicSystemManager< n_fields >::AlgebraicSystemManager(const MpiComm&       comm,
                                                           const MeshPartition& mesh,
                                                           ConstexprValue< problem_def >)
{
    initSystem(comm, mesh, ConstexprValue< problem_def >{});
}
template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
AlgebraicSystemManager< n_fields >::AlgebraicSystemManager(const MpiComm&       comm,
                                                           const MeshPartition& mesh,
                                                           ConstexprValue< problem_def >,
                                                           ConstexprValue< dirichlet_def >)
{
    const auto global_node_dof_map = initSystem(comm, mesh, ConstexprValue< problem_def >{});
    initDirichletBCs(mesh, global_node_dof_map, ConstexprValue< problem_def >{}, ConstexprValue< dirichlet_def >{});
}

template < size_t n_fields >
Teuchos::RCP< Tpetra::MultiVector< val_t, local_dof_t, global_dof_t > >
AlgebraicSystemManager< n_fields >::makeSolutionMultiVector(size_t n_cols) const
{
    return makeTeuchosRCP< mltvector_t >(m_sparsity_graph->getRowMap(), n_cols, false);
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::beginAssembly()
{
    switch (m_state)
    {
    case State::OpenForAssembly:
        return;
    case State::OpenForModify:
        throw std::runtime_error{
            "Initiation of assembly was attempted while the algebraic system was in the \"open for modification\" "
            "state. Finalize the modification first before calling \"beginAssembly\"."};
    case State::Closed:
        m_matrix->beginAssembly();
        m_rhs->beginAssembly();
        openRhs();
        m_state = State::OpenForAssembly;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::endAssembly()
{
    switch (m_state)
    {
    case State::OpenForAssembly:
        m_matrix->endAssembly();
        closeRhs();
        m_rhs->endAssembly();
        m_state = State::Closed;
        break;
    case State::OpenForModify:
        throw std::runtime_error{
            "Finilization of assembly was attempted while the algebraic system was in the \"open for modification\" "
            "state."};
    case State::Closed:
        return;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::beginModify()
{
    switch (m_state)
    {
    case State::OpenForAssembly:
        throw std::runtime_error{
            "Initiation of modification was attempted while the algebraic system was in the \"open for assembly\" "
            "state. Finalize the asembly first before calling \"beginModify\"."};
    case State::OpenForModify:
        return;
    case State::Closed:
        m_matrix->beginModify();
        m_rhs->beginModify();
        openRhs();
        m_state = State::OpenForModify;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::endModify()
{
    switch (m_state)
    {
    case State::OpenForAssembly:
        throw std::runtime_error{
            "Finilization of modification was attempted while the algebraic system was in the \"open for assembly\" "
            "state."};
    case State::OpenForModify:
        m_matrix->endModify();
        closeRhs();
        m_rhs->endModify();
        m_state = State::Closed;
        break;
    case State::Closed:
        return;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::setToZero()
{
    if (m_state == State::Closed)
        throw std::runtime_error{"The system may only be zeroed if it is in a non-closed state."};
    m_matrix->setAllToScalar(0.);
    m_rhs->putScalar(0.);
}

template < size_t n_fields >
template < BasisTypes              BT,
           QuadratureTypes         QT,
           q_o_t                   QO,
           ArrayOf_c< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter,
           detail::DomainIdRange_c  R >
void AlgebraicSystemManager< n_fields >::assembleDomainProblem(
    Kernel&& kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& fval_getter, val_t time)
{
    if (m_state == State::Closed)
        throw std::runtime_error{"Assemble was called while the algebraic system was in a closed state."};
    assembleGlobalSystem< BT, QT, QO, field_inds >(std::forward< Kernel >(kernel),
                                                   mesh,
                                                   std::forward< R >(domain_ids),
                                                   std::forward< FvalGetter >(fval_getter),
                                                   *m_matrix,
                                                   m_rhs_view,
                                                   m_node_to_row_dof_map,
                                                   m_node_to_col_dof_map,
                                                   m_node_to_rhs_dof_map,
                                                   time);
}

template < size_t n_fields >
template < BasisTypes              BT,
           QuadratureTypes         QT,
           q_o_t                   QO,
           ArrayOf_c< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter >
void AlgebraicSystemManager< n_fields >::assembleBoundaryProblem(Kernel&&            kernel,
                                                                 const BoundaryView& boundary,
                                                                 FvalGetter&&        fval_getter,
                                                                 val_t               time)
{
    if (m_state == State::Closed)
        throw std::runtime_error{"Assemble was called while the algebraic system was in a closed state."};
    assembleGlobalBoundarySystem< BT, QT, QO, field_inds >(std::forward< Kernel >(kernel),
                                                           boundary,
                                                           std::forward< FvalGetter >(fval_getter),
                                                           *m_matrix,
                                                           m_rhs_view,
                                                           m_node_to_row_dof_map,
                                                           m_node_to_col_dof_map,
                                                           m_node_to_rhs_dof_map,
                                                           time);
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::applyDirichletBCs(
    const Tpetra::Vector< val_t, local_dof_t, global_dof_t >& bc_vals)
{
    if (not m_dirichlet_bcs)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but no Dirichlet BCs were defined."};
    if (m_state != State::OpenForModify)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but the system was not in the \"open for "
                                 "modification\" state."};
    m_dirichlet_bcs->apply(bc_vals, *m_matrix, *m_rhs->getVectorNonConst(0));
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def >
NodeToGlobalDofMap< n_fields > AlgebraicSystemManager< n_fields >::initSystem(const MpiComm&                comm,
                                                                              const MeshPartition&          mesh,
                                                                              ConstexprValue< problem_def > problem)
{
    const auto dof_intervals       = computeDofIntervals(mesh, problem, comm);
    auto       node_global_dof_map = NodeToGlobalDofMap< n_fields >{mesh, dof_intervals};
    m_sparsity_graph               = detail::makeSparsityGraph(mesh, problem, dof_intervals, comm);
    m_matrix                       = makeTeuchosRCP< fematrix_t >(m_sparsity_graph);
    m_rhs = makeTeuchosRCP< fevector_t >(m_sparsity_graph->getRowMap(), m_sparsity_graph->getImporter(), 1u);
    m_node_to_row_dof_map = NodeToLocalDofMap< n_fields >{mesh, node_global_dof_map, *m_matrix->getRowMap()};
    m_node_to_col_dof_map = NodeToLocalDofMap< n_fields >{mesh, node_global_dof_map, *m_matrix->getColMap()};
    m_node_to_rhs_dof_map = NodeToLocalDofMap< n_fields >{mesh, node_global_dof_map, *m_rhs->getMap()};
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    m_state = State::OpenForAssembly;
    openRhs();
    return node_global_dof_map;
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
void AlgebraicSystemManager< n_fields >::initDirichletBCs(const MeshPartition&                  mesh,
                                                          const NodeToGlobalDofMap< n_fields >& dof_map,
                                                          ConstexprValue< problem_def >         problem,
                                                          ConstexprValue< dirichlet_def >       bcs)
{
    auto [owned_bcdofs, shared_bcdofs] = detail::getDirichletDofs(mesh, m_sparsity_graph, dof_map, problem, bcs);
    m_dirichlet_bcs.emplace(m_sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::openRhs()
{
    m_rhs->sync_host();
    m_rhs->modify_host();
    const auto rhs_alloc = m_rhs->getDataNonConst(0);
    m_rhs_view           = rhs_alloc; // Backed by m_rhs, this is not dangling
}
template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::closeRhs()
{
    m_rhs->sync_device();
    m_rhs_view = {};
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
