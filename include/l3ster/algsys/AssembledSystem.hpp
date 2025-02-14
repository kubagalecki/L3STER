#ifndef L3STER_ALGSYS_ASSEMBLEDSYSTEM_HPP
#define L3STER_ALGSYS_ASSEMBLEDSYSTEM_HPP

#include "l3ster/algsys/ComputeValuesAtNodes.hpp"
#include "l3ster/algsys/SparsityGraph.hpp"
#include "l3ster/algsys/StaticCondensationManager.hpp"
#include "l3ster/bcs/BCDefinition.hpp"
#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/post/FieldAccess.hpp"
#include "l3ster/solve/SolverInterface.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include <format>

namespace lstr::algsys
{
template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
class AssembledSystem
{
public:
    inline AssembledSystem(std::shared_ptr< const MpiComm >                          comm,
                           std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                           const ProblemDefinition< max_dofs_per_node >&             problem_def,
                           const BCDefinition< max_dofs_per_node >&                  bc_def);

    inline auto getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t >;
    inline auto getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >;
    inline auto getSolution() const -> Teuchos::RCP< tpetra_multivector_t >;

    inline void beginAssembly();
    inline void endAssembly();
    template < EquationKernel_c         Kernel,
               ArrayOf_c< size_t > auto field_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{} >
    void assembleProblem(const Kernel&                        kernel,
                         const util::ArrayOwner< d_id_t >&    domain_ids,
                         const post::FieldAccess< n_fields >& field_access      = {},
                         util::ConstexprValue< field_inds >   field_inds_ctwrpr = {},
                         util::ConstexprValue< asm_opts >     assembly_options  = {},
                         val_t                                time              = 0.)
        requires(Kernel::parameters.n_rhs == n_rhs);

    template < ResidualKernel_c Kernel, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setDirichletBCValues(const Kernel&                                                 kernel,
                              const util::ArrayOwner< d_id_t >&                             domain_ids,
                              const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
                              const post::FieldAccess< n_fields >&                          field_access = {},
                              val_t                                                         time         = 0.);
    template < size_t n_vals, std::integral dofind_t = size_t >
    void setDirichletBCValues(const std::array< val_t, n_vals >&    values,
                              const util::ArrayOwner< d_id_t >&     domain_ids,
                              const std::array< dofind_t, n_vals >& dof_inds)
        requires(n_rhs == 1);

    template < solvers::DirectSolver_c Solver >
    void solve(Solver& solver) const;
    template < solvers::IterativeSolver_c Solver >
    IterSolveResult solve(Solver& solver) const;

    inline void updateSolution(const util::ArrayOwner< size_t >& sol_inds,
                               SolutionManager&                  sol_man,
                               const util::ArrayOwner< size_t >& sol_man_inds);

    inline void describe(std::ostream& out = std::cout) const;

    template < ResidualKernel_c Kernel, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setValues(const Teuchos::RCP< tpetra_femultivector_t >&                 vector,
                   const Kernel&                                                 kernel,
                   const util::ArrayOwner< d_id_t >&                             domain_ids,
                   const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
                   const post::FieldAccess< n_fields >&                          field_access = {},
                   val_t                                                         time         = 0.) const;
    template < ResidualKernel_c Kernel, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setValues(const Teuchos::RCP< tpetra_multivector_t >&                   vector,
                   const Kernel&                                                 kernel,
                   const util::ArrayOwner< d_id_t >&                             domain_ids,
                   const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
                   const post::FieldAccess< n_fields >&                          field_access = {},
                   val_t                                                         time         = 0.) const;

private:
    inline auto initMultiVector(size_t cols = n_rhs) const -> Teuchos::RCP< tpetra_femultivector_t >;
    inline void setToZero();
    inline void applyDirichletBCs();

    enum struct State
    {
        OpenForAssembly,
        Closed
    };
    inline void assertState(State                expected,
                            std::string_view     err_msg,
                            std::source_location src_loc = std::source_location::current()) const;

    std::shared_ptr< const MpiComm >                          m_comm;
    std::shared_ptr< const mesh::MeshPartition< orders... > > m_mesh;
    Teuchos::RCP< tpetra_fecrsmatrix_t >                      m_matrix;
    Teuchos::RCP< tpetra_femultivector_t >                    m_rhs, m_solution;
    Teuchos::RCP< const tpetra_fecrsgraph_t >                 m_sparsity_graph;
    std::optional< const bcs::DirichletBCAlgebraic >          m_dirichlet_bcs;
    Teuchos::RCP< tpetra_femultivector_t >                    m_dirichlet_values;
    dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >           m_node_dof_map;
    StaticCondensationManager< CP >                           m_condensation_manager;
    State                                                     m_state;
};

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < solvers::DirectSolver_c Solver >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::solve(Solver& solver) const
{
    L3STER_PROFILE_FUNCTION;
    m_solution->switchActiveMultiVector();
    solver.solve(m_matrix, m_rhs, m_solution);
    m_solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    m_solution->switchActiveMultiVector();
#ifdef L3STER_PROFILE_EXECUTION
    m_solution->getMap()->getComm()->barrier();
#endif
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < solvers::IterativeSolver_c Solver >
IterSolveResult AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::solve(Solver& solver) const
{
    L3STER_PROFILE_FUNCTION;
    m_solution->switchActiveMultiVector();
    const auto solve_info = solver.solve(m_matrix, m_rhs, m_solution);
    m_solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    m_solution->switchActiveMultiVector();
#ifdef L3STER_PROFILE_EXECUTION
    m_solution->getMap()->getComm()->barrier();
#endif
    return solve_info;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::updateSolution(
    const util::ArrayOwner< size_t >& sol_inds,
    SolutionManager&                  sol_man,
    const util::ArrayOwner< size_t >& sol_man_inds)
{
    util::throwingAssert(sol_man_inds.size() == sol_inds.size() * n_rhs,
                         "Source and destination indices lengths must match");
    util::throwingAssert(std::ranges::none_of(sol_inds, [](size_t i) { return i >= max_dofs_per_node; }),
                         "Source index out of bounds");
    util::throwingAssert(std::ranges::none_of(sol_man_inds, [&](size_t i) { return i >= sol_man.nFields(); }),
                         "Destination index out of bounds");

    const auto solution_view = m_solution->getLocalViewHost(Tpetra::Access::ReadOnly);
    m_condensation_manager.recoverSolution(
        *m_mesh, m_node_dof_map, util::asSpans< n_rhs >(solution_view), sol_inds, sol_man, sol_man_inds);
}

namespace detail
{
template < std::invocable< const tpetra_femultivector_t::host_view_type&,
                           const tpetra_femultivector_t::host_view_type& > Compute >
void averageNodeVals(tpetra_femultivector_t& values, tpetra_femultivector_t& num_contribs, Compute&& compute)
{
    const auto num_rhs = values.getNumVectors();
    values.switchActiveMultiVector();
    const auto num_owned = values.getLocalLength();
    values.switchActiveMultiVector();
    const auto num_all = values.getLocalLength();
    values.beginAssembly();
    {
        num_contribs.beginAssembly();
        auto       own_contribs = util::DynamicBitset{num_owned};
        const auto vals_view    = values.getLocalViewHost(Tpetra::Access::ReadWrite);
        {
            const auto num_contribs_view = num_contribs.getLocalViewHost(Tpetra::Access::ReadWrite);
            for (size_t rhs = 0; rhs != num_rhs; ++rhs)
                for (size_t i = num_owned; i != num_all; ++i)
                    vals_view(i, rhs) = 0.;
            std::invoke(std::forward< Compute >(compute), vals_view, num_contribs_view);
            for (size_t i = 0; i != num_owned; ++i)
                own_contribs.assign(i, num_contribs_view(i, 0) > .5);
        }
        num_contribs.endAssembly();
        const auto num_contribs_view = num_contribs.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i != num_owned; ++i)
            if (not own_contribs.test(i) and num_contribs_view(i, 0) > .5)
                for (size_t rhs = 0; rhs != num_rhs; ++rhs)
                    vals_view(i, rhs) = 0.;
    }
    values.endAssembly();
    {
        const auto vals_view         = values.getLocalViewHost(Tpetra::Access::ReadWrite);
        const auto num_contribs_view = num_contribs.getLocalViewHost(Tpetra::Access::ReadOnly);
        for (size_t i = 0; i != num_owned; ++i)
        {
            const auto nc = num_contribs_view(i, 0);
            if (nc > 1.5)
                for (size_t rhs = 0; rhs != num_rhs; ++rhs)
                    vals_view(i, rhs) /= nc;
        }
    }
    values.doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    values.switchActiveMultiVector();
}
} // namespace detail

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < ResidualKernel_c Kernel, std::integral dofind_t, size_t n_fields >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::setValues(
    const Teuchos::RCP< tpetra_femultivector_t >&                 vector,
    const Kernel&                                                 kernel,
    const util::ArrayOwner< d_id_t >&                             domain_ids,
    const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
    const post::FieldAccess< n_fields >&                          field_access,
    val_t                                                         time) const
{
    util::throwingAssert(util::isValidIndexRange(dof_inds, max_dofs_per_node),
                         "The DOF indices are out of bounds for the problem");
    util::throwingAssert(n_rhs == vector->getNumVectors(), "The number of columns and number of RHS must match");

    auto       num_contribs       = initMultiVector(1);
    const auto compute_local_vals = [&](const tpetra_femultivector_t::host_view_type& vals_view,
                                        const tpetra_femultivector_t::host_view_type& num_contribs_view) {
        computeValuesAtNodes(
            kernel, *m_mesh, domain_ids, m_node_dof_map, dof_inds, field_access, vals_view, num_contribs_view, time);
    };
    detail::averageNodeVals(*vector, *num_contribs, compute_local_vals);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < ResidualKernel_c Kernel, std::integral dofind_t, size_t n_fields >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::setValues(
    const Teuchos::RCP< tpetra_multivector_t >&                   vector,
    const Kernel&                                                 kernel,
    const util::ArrayOwner< d_id_t >&                             domain_ids,
    const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
    const post::FieldAccess< n_fields >&                          field_access,
    val_t                                                         time) const
{
    const auto vector_downcast = Teuchos::rcp_dynamic_cast< tpetra_femultivector_t >(vector);
    setValues(vector_downcast, kernel, domain_ids, dof_inds, field_access, time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < ResidualKernel_c Kernel, std::integral dofind_t, size_t n_fields >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::setDirichletBCValues(
    const Kernel&                                                 kernel,
    const util::ArrayOwner< d_id_t >&                             domain_ids,
    const std::array< dofind_t, Kernel::parameters.n_equations >& dof_inds,
    const post::FieldAccess< n_fields >&                          field_access,
    val_t                                                         time)
{
    util::throwingAssert(m_dirichlet_bcs.has_value(), "setDirichletBCValues called, but no Dirichlet BCs were defined");
    util::throwingAssert(util::isValidIndexRange(dof_inds, max_dofs_per_node),
                         "The DOF indices are out of bounds for the problem");

    auto       num_contribs       = initMultiVector(1);
    const auto compute_local_vals = [&](const tpetra_femultivector_t::host_view_type& vals_view,
                                        const tpetra_femultivector_t::host_view_type& num_contribs_view) {
        computeValuesAtNodes(
            kernel, *m_mesh, domain_ids, m_node_dof_map, dof_inds, field_access, vals_view, num_contribs_view, time);
    };
    detail::averageNodeVals(*m_dirichlet_values, *num_contribs, compute_local_vals);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < size_t n_vals, std::integral dofind_t >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::setDirichletBCValues(
    const std::array< val_t, n_vals >&    values,
    const util::ArrayOwner< d_id_t >&     domain_ids,
    const std::array< dofind_t, n_vals >& dof_inds)
    requires(n_rhs == 1)
{
    util::throwingAssert(m_dirichlet_bcs.has_value(), "setDirichletBCValues called, but no Dirichlet BCs were defined");
    util::throwingAssert(util::isValidIndexRange(dof_inds, max_dofs_per_node),
                         "The DOF indices are out of bounds for the problem");

    const auto values_wrapped     = std::array{std::span{values}};
    auto       num_contribs       = initMultiVector(1);
    const auto compute_local_vals = [&](const tpetra_femultivector_t::host_view_type& vals_view,
                                        const tpetra_femultivector_t::host_view_type& num_contribs_view) {
        computeValuesAtNodes(
            *m_mesh, domain_ids, m_node_dof_map, dof_inds, values_wrapped, vals_view, num_contribs_view);
    };
    detail::averageNodeVals(*m_dirichlet_values, *num_contribs, compute_local_vals);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
auto AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::getMatrix() const
    -> Teuchos::RCP< const tpetra_crsmatrix_t >
{
    assertState(State::Closed, "`getMatrix()` was called before `endAssembly()`");
    return m_matrix;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
auto AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::getRhs() const
    -> Teuchos::RCP< const tpetra_multivector_t >
{
    assertState(State::Closed, "`getRhs()` was called before `endAssembly()`");
    return m_rhs;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
auto AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::getSolution() const
    -> Teuchos::RCP< tpetra_multivector_t >
{
    assertState(State::Closed, "`getSolution()` was called before `endAssembly()`");
    return m_solution;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
auto AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::initMultiVector(size_t cols) const
    -> Teuchos::RCP< tpetra_femultivector_t >
{
    return util::makeTeuchosRCP< tpetra_femultivector_t >(
        m_sparsity_graph->getRowMap(), m_sparsity_graph->getImporter(), cols);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::AssembledSystem(
    std::shared_ptr< const MpiComm >                          comm,
    std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
    const ProblemDefinition< max_dofs_per_node >&             problem_def,
    const BCDefinition< max_dofs_per_node >&                  bc_def)
    : m_comm{std::move(comm)}, m_mesh{std::move(mesh)}, m_state{State::OpenForAssembly}
{
    constexpr auto cp_tag              = CondensationPolicyTag< CP >{};
    const auto     periodic_bc         = bcs::PeriodicBC{bc_def.getPeriodic(), *m_mesh, *m_comm};
    const auto     node_global_dof_map = dofs::NodeToGlobalDofMap{*m_comm, *m_mesh, problem_def, periodic_bc, cp_tag};
    m_sparsity_graph                   = makeSparsityGraph(*m_comm, *m_mesh, node_global_dof_map, problem_def, cp_tag);

    L3STER_PROFILE_REGION_BEGIN("Create Tpetra objects");
    m_matrix   = util::makeTeuchosRCP< tpetra_fecrsmatrix_t >(m_sparsity_graph);
    m_rhs      = initMultiVector();
    m_solution = initMultiVector();
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    L3STER_PROFILE_REGION_END("Create Tpetra objects");

    m_node_dof_map =
        dofs::NodeToLocalDofMap{node_global_dof_map, *m_matrix->getRowMap(), *m_matrix->getColMap(), *m_rhs->getMap()};
    m_condensation_manager = StaticCondensationManager< CP >{*m_mesh, m_node_dof_map, problem_def, n_rhs};
    m_condensation_manager.beginAssembly();

    const auto& dirichlet = bc_def.getDirichlet();
    if (not dirichlet.empty())
    {
        L3STER_PROFILE_REGION_BEGIN("Dirichlet BCs");
        auto [owned_bcdofs, shared_bcdofs] =
            bcs::getDirichletDofs(*m_mesh, m_sparsity_graph, node_global_dof_map, dirichlet);
        m_dirichlet_bcs.emplace(m_sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
        L3STER_PROFILE_REGION_END("Dirichlet BCs");
    }
    if (m_dirichlet_bcs.has_value())
        m_dirichlet_values = initMultiVector();
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::beginAssembly()
{
    L3STER_PROFILE_FUNCTION;
    if (m_state == State::OpenForAssembly)
        return;

    m_state = State::OpenForAssembly;
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    m_condensation_manager.beginAssembly();
    setToZero();
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::endAssembly()
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`endAssembly()` was called more than once");

    L3STER_PROFILE_REGION_BEGIN("Static condensation");
    auto rhs_view_owner = m_rhs->getLocalViewHost(Tpetra::Access::ReadWrite);
    auto rhs_views      = util::asSpans< n_rhs >(rhs_view_owner);
    m_condensation_manager.endAssembly(*m_mesh, m_node_dof_map, *m_matrix, rhs_views);
    L3STER_PROFILE_REGION_END("Static condensation");
    L3STER_PROFILE_REGION_BEGIN("RHS");
    m_rhs->endAssembly();
    L3STER_PROFILE_REGION_END("RHS");
    L3STER_PROFILE_REGION_BEGIN("Matrix");
    m_matrix->endAssembly();
    L3STER_PROFILE_REGION_END("Matrix");
    m_state = State::Closed;

#ifdef L3STER_PROFILE_EXECUTION
    m_rhs->getMap()->getComm()->barrier();
#endif
    if (m_dirichlet_bcs.has_value())
        applyDirichletBCs();
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::setToZero()
{
    m_matrix->setAllToScalar(0.);
    m_rhs->putScalar(0.);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
template < EquationKernel_c Kernel, ArrayOf_c< size_t > auto field_inds, size_t n_fields, AssemblyOptions asm_opts >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::assembleProblem(
    const Kernel&                        kernel,
    const util::ArrayOwner< d_id_t >&    domain_ids,
    const post::FieldAccess< n_fields >& field_access,
    util::ConstexprValue< field_inds >   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >     assembly_options,
    val_t                                time)
    requires(Kernel::parameters.n_rhs == n_rhs)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`assembleProblem()` was called before `beginAssembly()`");
    auto rhs_view_owner = m_rhs->getLocalViewHost(Tpetra::Access::ReadWrite);
    auto rhs_views      = util::asSpans< n_rhs >(rhs_view_owner);
    assembleGlobalSystem(kernel,
                         *m_mesh,
                         domain_ids,
                         field_access,
                         *m_matrix,
                         rhs_views,
                         m_node_dof_map,
                         m_condensation_manager,
                         field_inds_ctwrpr,
                         assembly_options,
                         time);

#ifdef L3STER_PROFILE_EXECUTION
    m_rhs->getMap()->getComm()->barrier();
#endif
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::applyDirichletBCs()
{
    L3STER_PROFILE_FUNCTION;
    m_matrix->beginModify();
    m_rhs->beginModify();
    m_dirichlet_values->switchActiveMultiVector();
    m_dirichlet_bcs->apply(*m_dirichlet_values, *m_matrix, *m_rhs);
    m_dirichlet_values->switchActiveMultiVector();
    m_rhs->endModify();
    m_matrix->endModify();

#ifdef L3STER_PROFILE_EXECUTION
    m_rhs->getMap()->getComm()->barrier();
#endif
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::assertState(State                expected,
                                                                             std::string_view     err_msg,
                                                                             std::source_location src_loc) const
{
    util::throwingAssert(m_state == expected, err_msg, src_loc);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, size_t n_rhs, el_o_t... orders >
void AssembledSystem< max_dofs_per_node, CP, n_rhs, orders... >::describe(std::ostream& out) const
{
    const auto local_num_rows    = m_matrix->getLocalNumRows();
    const auto local_num_cols    = m_matrix->getLocalNumCols();
    const auto local_num_entries = m_matrix->getLocalNumEntries();
    auto       local_sizes_max   = std::array{local_num_rows, local_num_cols, local_num_entries};
    auto       local_sizes_min   = local_sizes_max;
    m_comm->reduceInPlace(local_sizes_max, 0, MPI_MAX);
    m_comm->reduceInPlace(local_sizes_min, 0, MPI_MIN);
    if (m_comm->getRank() == 0)
    {
        const auto global_num_rows_range = m_matrix->getRangeMap()->getGlobalNumElements();
        const auto global_num_rows_sum   = m_matrix->getGlobalNumRows();
        const auto global_num_cols       = m_matrix->getGlobalNumCols();
        const auto global_num_entries    = m_matrix->getGlobalNumEntries();
        out << std::format("The algebraic system has dimensions {} by {}\n"
                           "Distribution among {} MPI rank(s):\n"
                           "{:<10}|{:^17}|{:^17}|{:^17}|\n"
                           "{:<10}|{:^17}|{:^17}|{:^17}|\n"
                           "{:<10}|{:^17}|{:^17}|{:^17}|\n"
                           "{:<10}|{:^17}|{:^17}|{:^17}|\n\n",
                           global_num_rows_range,
                           global_num_cols,
                           m_comm->getSize(),
                           "",
                           "* MIN *",
                           "* MAX *",
                           "* TOTAL *",
                           "ROWS",
                           local_sizes_min[0],
                           local_sizes_max[0],
                           global_num_rows_sum,
                           "COLUMNS",
                           local_sizes_min[1],
                           local_sizes_max[1],
                           global_num_cols,
                           "NON-ZEROS",
                           local_sizes_min[2],
                           local_sizes_max[2],
                           global_num_entries);
    }
    m_comm->barrier();
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_ASSEMBLEDSYSTEM_HPP
