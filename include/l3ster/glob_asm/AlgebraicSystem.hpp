#ifndef L3STER_GLOB_ASM_ALGEBRAICSYSTEMMANAGER_HPP
#define L3STER_GLOB_ASM_ALGEBRAICSYSTEMMANAGER_HPP

#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/common/SparsityGraph.hpp"
#include "l3ster/glob_asm/ComputeValuesAtNodes.hpp"
#include "l3ster/glob_asm/StaticCondensationManager.hpp"
#include "l3ster/post/SolutionManager.hpp"
#include "l3ster/solve/SolverInterface.hpp"
#include "l3ster/util/Assertion.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/TypeID.hpp"
#include "l3ster/util/WeakCache.hpp"

namespace lstr
{
namespace glob_asm
{
template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
class AlgebraicSystem
{
    template < ProblemDef problem_def, ProblemDef dirichlet_def >
    AlgebraicSystem(const MpiComm&                                            comm,
                    std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                    util::ConstexprValue< problem_def >,
                    util::ConstexprValue< dirichlet_def >);

public:
    template < ProblemDef problem_def, ProblemDef dirichlet_def >
    static auto makeAlgebraicSystem(const MpiComm&                                            comm,
                                    std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                                    util::ConstexprValue< problem_def >                       problemdef_ctwrpr,
                                    util::ConstexprValue< dirichlet_def >                     dbcdef_ctwrpr)
        -> std::shared_ptr< AlgebraicSystem >;

    [[nodiscard]] inline auto getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t >;
    [[nodiscard]] inline auto getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >;
    [[nodiscard]] const auto& getDofMap() const { return m_node_dof_map; }

    [[nodiscard]] inline auto makeSolutionVector() const -> Teuchos::RCP< tpetra_femultivector_t >;
    template < IndexRange_c SolInds, IndexRange_c SolManInds >
    void updateSolution(const Teuchos::RCP< const tpetra_femultivector_t >& solution,
                        SolInds&&                                           sol_inds,
                        SolutionManager&                                    sol_man,
                        SolManInds&&                                        sol_man_inds);

    inline void beginAssembly();
    inline void endAssembly();

    template < typename Kernel,
               KernelParams             params,
               ArrayOf_c< size_t > auto field_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{} >
    void assembleDomainProblem(const DomainKernel< Kernel, params >&                kernel,
                               const util::ArrayOwner< d_id_t >&                    domain_ids,
                               const SolutionManager::FieldValueGetter< n_fields >& fval_getter       = {},
                               util::ConstexprValue< field_inds >                   field_inds_ctwrpr = {},
                               util::ConstexprValue< asm_opts >                     assembly_options  = {},
                               val_t                                                time              = 0.);
    template < typename Kernel,
               KernelParams             params,
               ArrayOf_c< size_t > auto field_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{} >
    void assembleBoundaryProblem(const BoundaryKernel< Kernel, params >&              kernel,
                                 const util::ArrayOwner< d_id_t >&                    boundary_ids,
                                 const SolutionManager::FieldValueGetter< n_fields >& fval_getter       = {},
                                 util::ConstexprValue< field_inds >                   field_inds_ctwrpr = {},
                                 util::ConstexprValue< asm_opts >                     assembly_options  = {},
                                 val_t                                                time              = 0.);

    inline void applyDirichletBCs();
    template < typename Kernel, KernelParams params, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setDirichletBCValues(const ResidualDomainKernel< Kernel, params >&        kernel,
                              const util::ArrayOwner< d_id_t >&                    domain_ids,
                              const std::array< dofind_t, params.n_equations >&    dof_inds,
                              const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = {},
                              val_t                                                time             = 0.);
    template < typename Kernel, KernelParams params, std::integral dofind_t = size_t, size_t n_fields = 0 >
    void setDirichletBCValues(const ResidualBoundaryKernel< Kernel, params >&      kernel,
                              const util::ArrayOwner< d_id_t >&                    boundary_ids,
                              const std::array< dofind_t, params.n_equations >&    dof_inds,
                              const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = {},
                              val_t                                                time             = 0.);

    void solve(solvers::Solver_c auto& solver, const Teuchos::RCP< tpetra_femultivector_t >& solution) const;

    inline void describe(const MpiComm& comm, std::ostream& out = std::cout) const;

private:
    inline void setToZero();

    inline auto getBoundaryView(const util::ArrayOwner< d_id_t >& ids) -> const mesh::BoundaryView< orders... >&;

    enum struct State
    {
        OpenForAssembly,
        Closed
    };
    inline void assertState(State                expected,
                            std::string_view     err_msg,
                            std::source_location src_loc = std::source_location::current()) const;

    std::shared_ptr< const mesh::MeshPartition< orders... > >               m_mesh;
    std::map< util::ArrayOwner< d_id_t >, mesh::BoundaryView< orders... > > m_boundary_views;
    Teuchos::RCP< tpetra_fecrsmatrix_t >                                    m_matrix;
    Teuchos::RCP< tpetra_femultivector_t >                                  m_rhs;
    Teuchos::RCP< const tpetra_fecrsgraph_t >                               m_sparsity_graph;
    std::optional< const bcs::DirichletBCAlgebraic >                        m_dirichlet_bcs;
    Teuchos::RCP< tpetra_multivector_t >                                    m_dirichlet_values;
    dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >                         m_node_dof_map;
    StaticCondensationManager< CP >                                         m_condensation_manager;
    State                                                                   m_state;

    // Caching mechanism to enable the reuse of the system allocation. For example, an adjoint problem will have the
    // same structure as the primal problem. We can therefore reuse the assembly data structures from the primal.
    // Note that this implies that assembly of different problems may not be interleaved, otherwise data will be
    // overwritten.
    //
    // The cache has weak ownership semantics, since we need all instances of Trilinos objects to be destroyed before
    // MPI and Kokkos are finalized.
    using cache_key_t      = std::pair< const void*, size_t >;
    using cache_key_hash_t = decltype([](const cache_key_t& key) { // hash quality is irrelevant here
        return std::hash< const void* >{}(key.first) ^ std::hash< size_t >{}(key.second);
    });
    static inline util::WeakCache< cache_key_t, AlgebraicSystem, cache_key_hash_t > cache{};
    template < ProblemDef problem_def, ProblemDef dirichlet_def >
    static auto makeCacheKey(const mesh::MeshPartition< orders... >& mesh,
                             util::ConstexprValue< problem_def >,
                             util::ConstexprValue< dirichlet_def >) -> cache_key_t
    {
        return std::make_pair(util::type_id_value< util::ValuePack< problem_def, dirichlet_def > >,
                              mesh.computeTopoHash());
    }
};

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
auto AlgebraicSystem< max_dofs_per_node, CP, orders... >::getBoundaryView(const util::ArrayOwner< d_id_t >& ids)
    -> const mesh::BoundaryView< orders... >&
{
    const auto existing_it = m_boundary_views.find(ids);
    if (existing_it != m_boundary_views.end())
        return existing_it->second;
    else
    {
        const auto [emplaced_it, _] = m_boundary_views.emplace(copy(ids), mesh::BoundaryView{*m_mesh, copy(ids)});
        return emplaced_it->second;
    }
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::solve(
    solvers::Solver_c auto& solver, const Teuchos::RCP< tpetra_femultivector_t >& solution) const
{
    L3STER_PROFILE_FUNCTION;
    solution->switchActiveMultiVector();
    solver.solve(m_matrix, m_rhs, solution);
    solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    solution->switchActiveMultiVector();
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < IndexRange_c SolInds, IndexRange_c SolManInds >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::updateSolution(
    const Teuchos::RCP< const tpetra_femultivector_t >& solution,
    SolInds&&                                           sol_inds,
    SolutionManager&                                    sol_man,
    SolManInds&&                                        sol_man_inds)
{
    util::throwingAssert(std::ranges::distance(sol_man_inds) == std::ranges::distance(sol_inds),
                         "Source and destination indices length must match");
    util::throwingAssert(std::ranges::none_of(sol_inds, [&](size_t i) { return i >= max_dofs_per_node; }),
                         "Source index out of bounds");
    util::throwingAssert(std::ranges::none_of(sol_man_inds, [&](size_t i) { return i >= sol_man.nFields(); }),
                         "Destination index out of bounds");

    const auto solution_view = Kokkos::subview(solution->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    m_condensation_manager.recoverSolution(*m_mesh,
                                           m_node_dof_map,
                                           util::asSpan(solution_view),
                                           std::forward< SolInds >(sol_inds),
                                           sol_man,
                                           std::forward< SolManInds >(sol_man_inds));
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < typename Kernel, KernelParams params, std::integral dofind_t, size_t n_fields >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::setDirichletBCValues(
    const ResidualDomainKernel< Kernel, params >&        kernel,
    const util::ArrayOwner< d_id_t >&                    domain_ids,
    const std::array< dofind_t, params.n_equations >&    dof_inds,
    const SolutionManager::FieldValueGetter< n_fields >& field_val_getter,
    val_t                                                time)
{
    util::throwingAssert(util::isValidIndexRange(dof_inds, max_dofs_per_node),
                         "The DOF indices are out of bounds for the problem");

    const auto vals_view =
        Kokkos::subview(m_dirichlet_values->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    computeValuesAtNodes(
        kernel, *m_mesh, domain_ids, m_node_dof_map, dof_inds, field_val_getter, util::asSpan(vals_view), time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < typename Kernel, KernelParams params, std::integral dofind_t, size_t n_fields >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::setDirichletBCValues(
    const ResidualBoundaryKernel< Kernel, params >&      kernel,
    const util::ArrayOwner< d_id_t >&                    boundary_ids,
    const std::array< dofind_t, params.n_equations >&    dof_inds,
    const SolutionManager::FieldValueGetter< n_fields >& field_val_getter,
    val_t                                                time)
{
    util::throwingAssert(util::isValidIndexRange(dof_inds, max_dofs_per_node),
                         "The DOF indices are out of bounds for the problem");

    const auto& boundary_view = getBoundaryView(boundary_ids);
    const auto  vals_view =
        Kokkos::subview(m_dirichlet_values->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    computeValuesAtBoundaryNodes(
        kernel, boundary_view, m_node_dof_map, dof_inds, field_val_getter, util::asSpan(vals_view), time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
auto AlgebraicSystem< max_dofs_per_node, CP, orders... >::getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t >
{
    assertState(State::Closed, "`getMatrix()` was called before `endAssembly()`");
    return m_matrix;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
auto AlgebraicSystem< max_dofs_per_node, CP, orders... >::getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >
{
    assertState(State::Closed, "`getRhs()` was called before `endAssembly()`");
    return m_rhs;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
auto AlgebraicSystem< max_dofs_per_node, CP, orders... >::makeSolutionVector() const
    -> Teuchos::RCP< tpetra_femultivector_t >
{
    return util::makeTeuchosRCP< tpetra_femultivector_t >(
        m_sparsity_graph->getRowMap(), m_sparsity_graph->getImporter(), 1u);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < ProblemDef problem_def, ProblemDef dirichlet_def >
auto AlgebraicSystem< max_dofs_per_node, CP, orders... >::makeAlgebraicSystem(
    const MpiComm&                                            comm,
    std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
    util::ConstexprValue< problem_def >                       problemdef_ctwrpr,
    util::ConstexprValue< dirichlet_def >                     dbcdef_ctwrpr) -> std::shared_ptr< AlgebraicSystem >
{
    auto       cache_key       = makeCacheKey(*mesh, problemdef_ctwrpr, dbcdef_ctwrpr);
    const char local_cache_hit = cache.contains(cache_key);
    char       global_cache_hit{};
    comm.allReduce(std::views::single(local_cache_hit), &global_cache_hit, MPI_LAND);
    if (global_cache_hit)
        return cache.get(cache_key);
    else
        return cache.emplace(cache_key, AlgebraicSystem{comm, std::move(mesh), problemdef_ctwrpr, dbcdef_ctwrpr});
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < ProblemDef problem_def, ProblemDef dirichlet_def >
AlgebraicSystem< max_dofs_per_node, CP, orders... >::AlgebraicSystem(
    const MpiComm&                                            comm,
    std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
    util::ConstexprValue< problem_def >                       problemdef_ctwrpr,
    util::ConstexprValue< dirichlet_def >                     dbcdef_ctwrpr)
    : m_mesh{std::move(mesh)}, m_state{State::OpenForAssembly}
{
    const auto cond_map            = dofs::makeCondensationMap< CP >(comm, *m_mesh, problemdef_ctwrpr);
    const auto dof_intervals       = computeDofIntervals(comm, *m_mesh, cond_map, problemdef_ctwrpr);
    const auto node_global_dof_map = dofs::NodeToGlobalDofMap{dof_intervals, cond_map};
    m_sparsity_graph               = makeSparsityGraph(comm, *m_mesh, node_global_dof_map, cond_map, problemdef_ctwrpr);

    L3STER_PROFILE_REGION_BEGIN("Create Tpetra objects");
    m_matrix = util::makeTeuchosRCP< tpetra_fecrsmatrix_t >(m_sparsity_graph);
    m_rhs    = makeSolutionVector();
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    L3STER_PROFILE_REGION_END("Create Tpetra objects");

    m_node_dof_map = dofs::NodeToLocalDofMap{
        cond_map, node_global_dof_map, *m_matrix->getRowMap(), *m_matrix->getColMap(), *m_rhs->getMap()};
    m_condensation_manager = StaticCondensationManager< CP >{*m_mesh, m_node_dof_map, problemdef_ctwrpr};
    m_condensation_manager.beginAssembly();

    if constexpr (dirichlet_def.n_domains != 0)
    {
        L3STER_PROFILE_REGION_BEGIN("Dirichlet BCs");
        auto [owned_bcdofs, shared_bcdofs] = bcs::getDirichletDofs(
            *m_mesh, m_sparsity_graph, node_global_dof_map, cond_map, problemdef_ctwrpr, dbcdef_ctwrpr);
        m_dirichlet_bcs.emplace(m_sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
        L3STER_PROFILE_REGION_END("Dirichlet BCs");
    }
    if (m_dirichlet_bcs.has_value())
        m_dirichlet_values = util::makeTeuchosRCP< tpetra_multivector_t >(m_sparsity_graph->getRowMap(), 1u);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::beginAssembly()
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

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::endAssembly()
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`endAssembly()` was called more than once");

    L3STER_PROFILE_REGION_BEGIN("Static condensation");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    m_condensation_manager.endAssembly(*m_mesh, m_node_dof_map, *m_matrix, util::asSpan(rhs_view));
    L3STER_PROFILE_REGION_END("Static condensation");
    L3STER_PROFILE_REGION_BEGIN("RHS");
    m_rhs->endAssembly();
    L3STER_PROFILE_REGION_END("RHS");
    L3STER_PROFILE_REGION_BEGIN("Matrix");
    m_matrix->endAssembly();
    L3STER_PROFILE_REGION_END("Matrix");
    m_state = State::Closed;
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::setToZero()
{
    m_matrix->setAllToScalar(0.);
    m_rhs->putScalar(0.);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < typename Kernel,
           KernelParams             params,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   n_fields,
           AssemblyOptions          asm_opts >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::assembleDomainProblem(
    const DomainKernel< Kernel, params >&                kernel,
    const util::ArrayOwner< d_id_t >&                    domain_ids,
    const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
    util::ConstexprValue< field_inds >                   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >                     assembly_options,
    val_t                                                time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`assembleDomainProblem()` was called before `beginAssembly()`");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    assembleGlobalSystem(kernel,
                         *m_mesh,
                         domain_ids,
                         fval_getter,
                         *m_matrix,
                         util::asSpan(rhs_view),
                         getDofMap(),
                         m_condensation_manager,
                         field_inds_ctwrpr,
                         assembly_options,
                         time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
template < typename Kernel,
           KernelParams             params,
           ArrayOf_c< size_t > auto field_inds,
           size_t                   n_fields,
           AssemblyOptions          asm_opts >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::assembleBoundaryProblem(
    const BoundaryKernel< Kernel, params >&              kernel,
    const util::ArrayOwner< d_id_t >&                    boundary_ids,
    const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
    util::ConstexprValue< field_inds >                   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >                     assembly_options,
    val_t                                                time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`assembleBoundaryProblem()` was called before `beginAssembly()`");
    const auto& boundary = getBoundaryView(boundary_ids);
    const auto  rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    assembleGlobalBoundarySystem(kernel,
                                 boundary,
                                 fval_getter,
                                 *m_matrix,
                                 util::asSpan(rhs_view),
                                 getDofMap(),
                                 m_condensation_manager,
                                 field_inds_ctwrpr,
                                 assembly_options,
                                 time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::applyDirichletBCs()
{
    L3STER_PROFILE_FUNCTION;
    util::throwingAssert(m_dirichlet_bcs.has_value(),
                         "`applyDirichletBCs()` was called, but no Dirichlet BCs were defined");
    assertState(State::Closed, "`applyDirichletBCs()` was called before `endAssembly()`");
    m_matrix->beginModify();
    m_rhs->beginModify();
    m_dirichlet_bcs->apply(*m_dirichlet_values->getVector(0), *m_matrix, *m_rhs->getVectorNonConst(0));
    m_rhs->endModify();
    m_matrix->endModify();
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::assertState(State                expected,
                                                                      std::string_view     err_msg,
                                                                      std::source_location src_loc) const
{
    util::throwingAssert(m_state == expected, err_msg, src_loc);
}

template < size_t max_dofs_per_node, CondensationPolicy CP, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP, orders... >::describe(const MpiComm& comm, std::ostream& out) const
{
    const auto local_num_rows    = m_matrix->getLocalNumRows();
    const auto local_num_cols    = m_matrix->getLocalNumCols();
    const auto local_num_entries = m_matrix->getLocalNumEntries();
    auto       local_sizes_max   = std::array{local_num_rows, local_num_cols, local_num_entries};
    auto       local_sizes_min   = local_sizes_max;
    comm.reduceInPlace(local_sizes_max, 0, MPI_MAX);
    comm.reduceInPlace(local_sizes_min, 0, MPI_MIN);
    if (comm.getRank() == 0)
    {
        const auto        global_num_rows_range = m_matrix->getRangeMap()->getGlobalNumElements();
        const auto        global_num_rows_sum   = m_matrix->getGlobalNumRows();
        const auto        global_num_cols       = m_matrix->getGlobalNumCols();
        const auto        global_num_entries    = m_matrix->getGlobalNumEntries();
        std::stringstream msg;
        msg << "\nThe algebraic system has dimensions " << global_num_rows_range << " by " << global_num_cols
            << "\nDistribution among MPI ranks (min, max, total):"
            << "\nRows:             " << local_sizes_min[0] << ", " << local_sizes_max[0] << ", " << global_num_rows_sum
            << "\nColumns:          " << local_sizes_min[1] << ", " << local_sizes_max[1] << ", " << global_num_cols
            << "\nNon-zero entries: " << local_sizes_min[2] << ", " << local_sizes_max[2] << ", " << global_num_entries
            << "\n\n";
        out << msg.view();
    }
    comm.barrier();
}
} // namespace glob_asm

template < el_o_t... orders,
           CondensationPolicy CP,
           ProblemDef         problem_def,
           ProblemDef         dirichlet_def = ProblemDef< 0, problem_def.n_fields >{} >
auto makeAlgebraicSystem(const MpiComm&                                            comm,
                         std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                         CondensationPolicyTag< CP >                             = {},
                         util::ConstexprValue< problem_def >   problemdef_ctwrpr = {},
                         util::ConstexprValue< dirichlet_def > dbcdef_ctwrpr     = {})
    -> std::shared_ptr< glob_asm::AlgebraicSystem< problem_def.n_fields, CP, orders... > >
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto max_dofs_per_node = problem_def.n_fields;
    return glob_asm::AlgebraicSystem< max_dofs_per_node, CP, orders... >::makeAlgebraicSystem(
        comm, std::move(mesh), problemdef_ctwrpr, dbcdef_ctwrpr);
}

template < el_o_t... orders,
           CondensationPolicy CP,
           ProblemDef         problem_def,
           ProblemDef         dirichlet_def = ProblemDef< 0, problem_def.n_fields >{} >
auto makeAlgebraicSystem(const MpiComm&                                      comm,
                         std::shared_ptr< mesh::MeshPartition< orders... > > mesh,
                         CondensationPolicyTag< CP >                         cond_policy_tag,
                         util::ConstexprValue< problem_def >                 problemdef_ctwrpr,
                         util::ConstexprValue< dirichlet_def >               dbcdef_ctwrpr = {})
    -> std::shared_ptr< glob_asm::AlgebraicSystem< problem_def.n_fields, CP, orders... > >
{
    return makeAlgebraicSystem<>(comm,
                                 std::shared_ptr< const mesh::MeshPartition< orders... > >{std::move(mesh)},
                                 cond_policy_tag,
                                 problemdef_ctwrpr,
                                 dbcdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_GLOB_ASM_ALGEBRAICSYSTEMMANAGER_HPP
