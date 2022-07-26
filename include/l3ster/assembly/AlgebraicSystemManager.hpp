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
public:
    template < detail::ProblemDef_c auto problem_def >
    AlgebraicSystemManager(const MpiComm& comm, const MeshPartition& mesh, ConstexprValue< problem_def >);
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    AlgebraicSystemManager(const MpiComm&       comm,
                           const MeshPartition& mesh,
                           ConstexprValue< problem_def >,
                           ConstexprValue< dirichlet_def >);

    const auto& getNodeToDofMap() const { return node_dof_map; }
    const auto& getMatrix() const { return matrix; }
    const auto& getRhs() const { return rhs; }

    inline Teuchos::RCP< Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t > >
    makeCompatibleFEMultiVector(size_t n_cols) const;
    inline Teuchos::RCP< Tpetra::MultiVector< val_t, local_dof_t, global_dof_t > >
    makeCompatibleMultiVector(size_t n_cols) const;

    inline void beginAssembly();
    inline void endAssembly();
    inline void beginModify();
    inline void endModify();
    inline void setToZero();

    template < BasisTypes              BT,
               QuadratureTypes         QT,
               q_o_t                   QO,
               array_of< size_t > auto field_inds,
               typename Kernel,
               detail::FieldValGetter_c FvalGetter,
               detail::DomainIdRange_c  R >
    void assembleDomainProblem(
        Kernel&& kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& fval_getter, val_t time = 0.);
    template < BasisTypes              BT,
               QuadratureTypes         QT,
               q_o_t                   QO,
               array_of< size_t > auto field_inds,
               typename Kernel,
               detail::FieldValGetter_c FvalGetter >
    void
    assembleBoundaryProblem(Kernel&& kernel, const BoundaryView& boundary, FvalGetter&& fval_getter, val_t time = 0.);

    inline void applyDirichletBCs(const Tpetra::Vector< val_t, local_dof_t, global_dof_t >& bc_vals);

private:
    template < detail::ProblemDef_c auto problem_def >
    void initSystem(const MpiComm& comm, const MeshPartition& mesh, ConstexprValue< problem_def >);
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    void initDirichletBCs(const MeshPartition& mesh, ConstexprValue< problem_def >, ConstexprValue< dirichlet_def >);
    inline void openRhs();
    inline void closeRhs();

    enum class State
    {
        OpenForAssembly,
        OpenForModify,
        Closed
    };

    NodeToDofMap< n_fields >                                                  node_dof_map;
    Teuchos::RCP< const Tpetra::Map< local_dof_t, global_dof_t > >            row_dist_map_openasm;
    Teuchos::RCP< Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t > >   matrix;
    Teuchos::RCP< Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t > > rhs;
    Teuchos::ArrayRCP< val_t >                                                rhs_raw_view;
    Teuchos::RCP< const Tpetra::FECrsGraph< local_dof_t, global_dof_t > >     sparsity_graph;
    std::optional< DirichletBCAlgebraic >                                     dirichlet_bcs;
    State                                                                     state;
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
    initSystem(comm, mesh, ConstexprValue< problem_def >{});
    initDirichletBCs(mesh, ConstexprValue< problem_def >{}, ConstexprValue< dirichlet_def >{});
}

template < size_t n_fields >
Teuchos::RCP< Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t > >
AlgebraicSystemManager< n_fields >::makeCompatibleFEMultiVector(size_t n_cols) const
{
    return makeTeuchosRCP< Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t > >(
        sparsity_graph->getRowMap(), sparsity_graph->getImporter(), n_cols);
}
template < size_t n_fields >
Teuchos::RCP< Tpetra::MultiVector< val_t, local_dof_t, global_dof_t > >
AlgebraicSystemManager< n_fields >::makeCompatibleMultiVector(size_t n_cols) const
{
    return makeTeuchosRCP< Tpetra::MultiVector< val_t, local_dof_t, global_dof_t > >(rhs->getMap(), n_cols);
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::beginAssembly()
{
    switch (state)
    {
    case State::OpenForAssembly:
        return;
    case State::OpenForModify:
        throw std::runtime_error{
            "Initiation of assembly was attempted while the algebraic system was in the \"open for modification\" "
            "state. Finalize the modification first before calling \"beginAssembly\"."};
    case State::Closed:
        matrix->beginAssembly();
        rhs->beginAssembly();
        openRhs();
        state = State::OpenForAssembly;
    }
}
template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::endAssembly()
{
    switch (state)
    {
    case State::OpenForAssembly:
        matrix->endAssembly();
        closeRhs();
        rhs->endAssembly();
        state = State::Closed;
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
    switch (state)
    {
    case State::OpenForAssembly:
        throw std::runtime_error{
            "Initiation of modification was attempted while the algebraic system was in the \"open for assembly\" "
            "state. Finalize the asembly first before calling \"beginModify\"."};
    case State::OpenForModify:
        return;
    case State::Closed:
        matrix->beginModify();
        rhs->beginModify();
        openRhs();
        state = State::OpenForModify;
    }
}
template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::endModify()
{
    switch (state)
    {
    case State::OpenForAssembly:
        throw std::runtime_error{
            "Finilization of modification was attempted while the algebraic system was in the \"open for assembly\" "
            "state."};
    case State::OpenForModify:
        matrix->endModify();
        closeRhs();
        rhs->endModify();
        state = State::Closed;
        break;
    case State::Closed:
        return;
    }
}
template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::setToZero()
{
    if (state == State::Closed)
        throw std::runtime_error{"The system may only be zeroed if it is in a non-closed state."};
    matrix->setAllToScalar(0.);
    rhs->putScalar(0.);
}

template < size_t n_fields >
template < BasisTypes              BT,
           QuadratureTypes         QT,
           q_o_t                   QO,
           array_of< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter,
           detail::DomainIdRange_c  R >
void AlgebraicSystemManager< n_fields >::assembleDomainProblem(
    Kernel&& kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& fval_getter, val_t time)
{
    if (state == State::Closed)
        throw std::runtime_error{"Assemble was called while the algebraic system was in a closed state."};
    assembleGlobalSystem< BT, QT, QO, field_inds >(std::forward< Kernel >(kernel),
                                                   mesh,
                                                   std::forward< R >(domain_ids),
                                                   node_dof_map,
                                                   std::forward< FvalGetter >(fval_getter),
                                                   *matrix,
                                                   rhs_raw_view,
                                                   *row_dist_map_openasm,
                                                   time);
}

template < size_t n_fields >
template < BasisTypes              BT,
           QuadratureTypes         QT,
           q_o_t                   QO,
           array_of< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter >
void AlgebraicSystemManager< n_fields >::assembleBoundaryProblem(Kernel&&            kernel,
                                                                 const BoundaryView& boundary,
                                                                 FvalGetter&&        fval_getter,
                                                                 val_t               time)
{
    if (state == State::Closed)
        throw std::runtime_error{"Assemble was called while the algebraic system was in a closed state."};
    assembleGlobalBoundarySystem< BT, QT, QO, field_inds >(std::forward< Kernel >(kernel),
                                                           boundary,
                                                           node_dof_map,
                                                           std::forward< FvalGetter >(fval_getter),
                                                           *matrix,
                                                           rhs_raw_view,
                                                           *row_dist_map_openasm,
                                                           time);
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::applyDirichletBCs(
    const Tpetra::Vector< val_t, local_dof_t, global_dof_t >& bc_vals)
{
    if (not dirichlet_bcs)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but no Dirichlet BCs were defined."};
    if (state != State::OpenForModify)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but the system was not in the \"open for "
                                 "modification\" state."};
    dirichlet_bcs->apply(bc_vals, *matrix, *rhs->getVectorNonConst(0));
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def >
void AlgebraicSystemManager< n_fields >::initSystem(const MpiComm&       comm,
                                                    const MeshPartition& mesh,
                                                    ConstexprValue< problem_def >)
{
    const auto dof_intervals = computeDofIntervals(mesh, ConstexprValue< problem_def >{}, comm);
    node_dof_map             = NodeToDofMap< n_fields >{mesh, dof_intervals};
    sparsity_graph           = detail::makeSparsityGraph(mesh, ConstexprValue< problem_def >{}, dof_intervals, comm);
    matrix = makeTeuchosRCP< Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t > >(sparsity_graph);
    rhs    = makeCompatibleFEMultiVector(1u);
    matrix->beginAssembly();
    rhs->beginAssembly();
    state                = State::OpenForAssembly;
    row_dist_map_openasm = rhs->getMap();
    openRhs();
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
void AlgebraicSystemManager< n_fields >::initDirichletBCs(const MeshPartition& mesh,
                                                          ConstexprValue< problem_def >,
                                                          ConstexprValue< dirichlet_def >)
{
    auto [owned_bcdofs, shared_bcdofs] = detail::getDirichletDofs(
        mesh, sparsity_graph, node_dof_map, ConstexprValue< problem_def >{}, ConstexprValue< dirichlet_def >{});
    dirichlet_bcs.emplace(sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::openRhs()
{
    rhs_raw_view = rhs->get1dViewNonConst();
    rhs->sync_host();
    rhs->modify_host();
}
template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::closeRhs()
{
    rhs->sync_device();
    rhs_raw_view = Teuchos::ArrayRCP< val_t >{};
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
