#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP

#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/assembly/StaticCondensationManager.hpp"
#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/solve/SolverInterface.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/SourceLocation.hpp"
#include "l3ster/util/TypeID.hpp"
#include "l3ster/util/WeakCache.hpp"

namespace lstr
{
template < size_t max_dofs_per_node, CondensationPolicy CP >
class AlgebraicSystem
{
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    AlgebraicSystem(const MpiComm&       comm,
                    const MeshPartition& mesh,
                    ConstexprValue< problem_def >,
                    ConstexprValue< dirichlet_def >);

public:
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    static auto makeAlgebraicSystem(const MpiComm&                  comm,
                                    const MeshPartition&            mesh,
                                    ConstexprValue< problem_def >   problemdef_ctwrpr,
                                    ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
        -> std::shared_ptr< AlgebraicSystem >;

    [[nodiscard]] inline auto getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t >;
    [[nodiscard]] inline auto getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >;
    [[nodiscard]] const auto& getDofMap() const { return m_node_dof_map; }

    [[nodiscard]] inline auto makeSolutionVector() const -> Teuchos::RCP< tpetra_femultivector_t >;
    void                      updateSolution(const MeshPartition&                                mesh,
                                             const Teuchos::RCP< const tpetra_femultivector_t >& solution,
                                             IndexRange_c auto&&                                 sol_inds,
                                             SolutionManager&                                    sol_man,
                                             IndexRange_c auto&&                                 sol_man_inds);

    inline void beginAssembly();
    inline void endAssembly(const MeshPartition& mesh);

    template < ArrayOf_c< size_t > auto field_inds = makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{} >
    void assembleDomainProblem(auto&&                                               kernel,
                               const MeshPartition&                                 mesh,
                               detail::DomainIdRange_c auto&&                       domain_ids,
                               const SolutionManager::FieldValueGetter< n_fields >& fval_getter       = {},
                               ConstexprValue< field_inds >                         field_inds_ctwrpr = {},
                               ConstexprValue< asm_opts >                           assembly_options  = {},
                               val_t                                                time              = 0.);
    template < ArrayOf_c< size_t > auto field_inds = makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{} >
    void assembleBoundaryProblem(auto&&                                               kernel,
                                 const BoundaryView&                                  boundary,
                                 const SolutionManager::FieldValueGetter< n_fields >& fval_getter       = {},
                                 ConstexprValue< field_inds >                         field_inds_ctwrpr = {},
                                 ConstexprValue< asm_opts >                           assembly_options  = {},
                                 val_t                                                time              = 0.);

    inline void applyDirichletBCs();
    template < IndexRange_c auto dof_inds = makeIotaArray< size_t, max_dofs_per_node >(), size_t n_fields = 0 >
    void setDirichletBCValues(auto&&                                               kernel,
                              const MeshPartition&                                 mesh,
                              detail::DomainIdRange_c auto&&                       domain_ids,
                              ConstexprValue< dof_inds >                           dofinds_ctwrpr   = {},
                              const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = {},
                              val_t                                                time             = 0.);
    template < IndexRange_c auto dof_inds = makeIotaArray< size_t, max_dofs_per_node >(), size_t n_fields = 0 >
    void setDirichletBCValues(auto&&                                               kernel,
                              const BoundaryView&                                  mesh,
                              ConstexprValue< dof_inds >                           dofinds_ctwrpr   = {},
                              const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = 0,
                              val_t                                                time             = 0.);

    void solve(Solver_c auto& solver, const Teuchos::RCP< tpetra_femultivector_t >& solution) const;

private:
    inline void setToZero();

    enum struct State
    {
        OpenForAssembly,
        Closed
    };
    inline void assertState(State                expected,
                            std::string_view     err_msg,
                            std::source_location src_loc = std::source_location::current()) const;

    Teuchos::RCP< tpetra_fecrsmatrix_t >        m_matrix;
    Teuchos::RCP< tpetra_femultivector_t >      m_rhs;
    Teuchos::RCP< const tpetra_fecrsgraph_t >   m_sparsity_graph;
    std::optional< const DirichletBCAlgebraic > m_dirichlet_bcs;
    Teuchos::RCP< tpetra_multivector_t >        m_dirichlet_values;
    NodeToLocalDofMap< max_dofs_per_node, 3 >   m_node_dof_map;
    detail::StaticCondensationManager< CP >     m_condensation_manager;
    State                                       m_state;

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
    static inline WeakCache< cache_key_t, AlgebraicSystem, cache_key_hash_t > cache{};
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    static auto makeCacheKey(const MeshPartition& mesh, ConstexprValue< problem_def >, ConstexprValue< dirichlet_def >)
        -> cache_key_t
    {
        return std::make_pair(type_id_value< ValuePack< problem_def, dirichlet_def > >, mesh.computeTopoHash());
    }
};

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::solve(Solver_c auto&                                solver,
                                                     const Teuchos::RCP< tpetra_femultivector_t >& solution) const
{
    solution->switchActiveMultiVector();
    solver.solve(m_matrix, m_rhs, solution);
    solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    solution->switchActiveMultiVector();
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::updateSolution(
    const MeshPartition&                                mesh,
    const Teuchos::RCP< const tpetra_femultivector_t >& solution,
    IndexRange_c auto&&                                 sol_inds,
    SolutionManager&                                    sol_man,
    IndexRange_c auto&&                                 sol_man_inds)
{
    util::throwingAssert(std::ranges::distance(sol_man_inds) == std::ranges::distance(sol_inds),
                         "Source and destination indices length must match");
    util::throwingAssert(std::ranges::none_of(sol_inds, [&](size_t i) { return i >= max_dofs_per_node; }),
                         "Source index out of bounds");
    util::throwingAssert(std::ranges::none_of(sol_man_inds, [&](size_t i) { return i >= sol_man.nFields(); }),
                         "Destination index out of bounds");

    const auto solution_view = Kokkos::subview(solution->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    m_condensation_manager.recoverSolution(mesh,
                                           m_node_dof_map,
                                           asSpan(solution_view),
                                           std::forward< decltype(sol_inds) >(sol_inds),
                                           sol_man,
                                           std::forward< decltype(sol_man_inds) >(sol_man_inds));
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < IndexRange_c auto dof_inds, size_t n_fields >
void AlgebraicSystem< max_dofs_per_node, CP >::setDirichletBCValues(
    auto&&                                               kernel,
    const MeshPartition&                                 mesh,
    detail::DomainIdRange_c auto&&                       domain_ids,
    ConstexprValue< dof_inds >                           dofinds_ctwrpr,
    const SolutionManager::FieldValueGetter< n_fields >& field_val_getter,
    val_t                                                time)
{
    const auto vals_view =
        Kokkos::subview(m_dirichlet_values->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    computeValuesAtNodes(std::forward< decltype(kernel) >(kernel),
                         mesh,
                         std::forward< decltype(domain_ids) >(domain_ids),
                         m_node_dof_map,
                         dofinds_ctwrpr,
                         std::forward< decltype(field_val_getter) >(field_val_getter),
                         asSpan(vals_view),
                         time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < IndexRange_c auto dof_inds, size_t n_fields >
void AlgebraicSystem< max_dofs_per_node, CP >::setDirichletBCValues(
    auto&&                                               kernel,
    const BoundaryView&                                  boundary_view,
    ConstexprValue< dof_inds >                           dofinds_ctwrpr,
    const SolutionManager::FieldValueGetter< n_fields >& field_val_getter,
    val_t                                                time)
{
    const auto vals_view =
        Kokkos::subview(m_dirichlet_values->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    computeValuesAtNodes(std::forward< decltype(kernel) >(kernel),
                         boundary_view,
                         m_node_dof_map,
                         dofinds_ctwrpr,
                         std::forward< decltype(field_val_getter) >(field_val_getter),
                         asSpan(vals_view),
                         time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
auto AlgebraicSystem< max_dofs_per_node, CP >::getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t >
{
    assertState(State::Closed, "`getMatrix()` was called before `endAssembly()`");
    return m_matrix;
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
auto AlgebraicSystem< max_dofs_per_node, CP >::getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >
{
    assertState(State::Closed, "`getRhs()` was called before `endAssembly()`");
    return m_rhs;
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
auto AlgebraicSystem< max_dofs_per_node, CP >::makeSolutionVector() const -> Teuchos::RCP< tpetra_femultivector_t >
{
    return makeTeuchosRCP< tpetra_femultivector_t >(m_sparsity_graph->getRowMap(), m_sparsity_graph->getImporter(), 1u);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
auto AlgebraicSystem< max_dofs_per_node, CP >::makeAlgebraicSystem(const MpiComm&                  comm,
                                                                   const MeshPartition&            mesh,
                                                                   ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                                   ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
    -> std::shared_ptr< AlgebraicSystem >
{
    auto       cache_key       = makeCacheKey(mesh, problemdef_ctwrpr, dbcdef_ctwrpr);
    const char local_cache_hit = cache.contains(cache_key);
    char       global_cache_hit{};
    comm.allReduce(std::views::single(local_cache_hit), &global_cache_hit, MPI_LAND);
    if (global_cache_hit)
        return cache.get(cache_key);
    else
        return cache.emplace(cache_key, AlgebraicSystem{comm, mesh, problemdef_ctwrpr, dbcdef_ctwrpr});
}

template < detail::ProblemDef_c auto problem_def, CondensationPolicy CP >
AlgebraicSystem(const MpiComm&, const MeshPartition&, ConstexprValue< problem_def >, CondensationPolicyTag< CP >)
    -> AlgebraicSystem< detail::deduceNFields(problem_def), CP >;
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def, CondensationPolicy CP >
AlgebraicSystem(const MpiComm&,
                const MeshPartition&,
                ConstexprValue< problem_def >,
                ConstexprValue< dirichlet_def >,
                CondensationPolicyTag< CP >) -> AlgebraicSystem< detail::deduceNFields(problem_def), CP >;

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
AlgebraicSystem< max_dofs_per_node, CP >::AlgebraicSystem(const MpiComm&                  comm,
                                                          const MeshPartition&            mesh,
                                                          ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                          ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
    : m_state{State::OpenForAssembly}
{
    const auto cond_map            = detail::makeCondensationMap< CP >(comm, mesh, problemdef_ctwrpr);
    const auto dof_intervals       = computeDofIntervals(comm, mesh, cond_map, problemdef_ctwrpr);
    const auto node_global_dof_map = NodeToGlobalDofMap{dof_intervals, cond_map};
    m_sparsity_graph = detail::makeSparsityGraph(comm, mesh, node_global_dof_map, cond_map, problemdef_ctwrpr);

    L3STER_PROFILE_REGION_BEGIN("Create Tpetra objects");
    m_matrix = makeTeuchosRCP< tpetra_fecrsmatrix_t >(m_sparsity_graph);
    m_rhs    = makeSolutionVector();
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    L3STER_PROFILE_REGION_END("Create Tpetra objects");

    m_node_dof_map = NodeToLocalDofMap{
        cond_map, node_global_dof_map, *m_matrix->getRowMap(), *m_matrix->getColMap(), *m_rhs->getMap()};
    m_condensation_manager = detail::StaticCondensationManager< CP >{mesh, m_node_dof_map, problemdef_ctwrpr};

    if constexpr (dirichlet_def.size() != 0)
    {
        L3STER_PROFILE_REGION_BEGIN("Dirichlet BCs");
        auto [owned_bcdofs, shared_bcdofs] = detail::getDirichletDofs(
            mesh, m_sparsity_graph, node_global_dof_map, cond_map, problemdef_ctwrpr, dbcdef_ctwrpr);
        m_dirichlet_bcs.emplace(m_sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
        L3STER_PROFILE_REGION_END("Dirichlet BCs");
    }
    if (m_dirichlet_bcs.has_value())
        m_dirichlet_values = makeTeuchosRCP< tpetra_multivector_t >(m_sparsity_graph->getRowMap(), 1u);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::beginAssembly()
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

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::endAssembly(const MeshPartition& mesh)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`endAssembly()` was called more than once");

    L3STER_PROFILE_REGION_BEGIN("Static condensation");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    m_condensation_manager.endAssembly(mesh, m_node_dof_map, *m_matrix, asSpan(rhs_view));
    L3STER_PROFILE_REGION_END("Static condensation");
    L3STER_PROFILE_REGION_BEGIN("RHS");
    m_rhs->endAssembly();
    L3STER_PROFILE_REGION_END("RHS");
    L3STER_PROFILE_REGION_BEGIN("Matrix");
    m_matrix->endAssembly();
    L3STER_PROFILE_REGION_END("Matrix");
    m_state = State::Closed;
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::setToZero()
{
    m_matrix->setAllToScalar(0.);
    m_rhs->putScalar(0.);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < ArrayOf_c< size_t > auto field_inds, size_t n_fields, AssemblyOptions asm_opts >
void AlgebraicSystem< max_dofs_per_node, CP >::assembleDomainProblem(
    auto&&                                               kernel,
    const MeshPartition&                                 mesh,
    detail::DomainIdRange_c auto&&                       domain_ids,
    const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
    ConstexprValue< field_inds >                         field_inds_ctwrpr,
    ConstexprValue< asm_opts >                           assembly_options,
    val_t                                                time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`assembleDomainProblem()` was called before `beginAssembly()`");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    assembleGlobalSystem(std::forward< decltype(kernel) >(kernel),
                         mesh,
                         std::forward< decltype(domain_ids) >(domain_ids),
                         std::forward< decltype(fval_getter) >(fval_getter),
                         *m_matrix,
                         asSpan(rhs_view),
                         getDofMap(),
                         m_condensation_manager,
                         field_inds_ctwrpr,
                         assembly_options,
                         time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < ArrayOf_c< size_t > auto field_inds, size_t n_fields, AssemblyOptions asm_opts >
void AlgebraicSystem< max_dofs_per_node, CP >::assembleBoundaryProblem(
    auto&&                                               kernel,
    const BoundaryView&                                  boundary,
    const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
    ConstexprValue< field_inds >                         field_inds_ctwrpr,
    ConstexprValue< asm_opts >                           assembly_options,
    val_t                                                time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`assembleBoundaryProblem()` was called before `beginAssembly()`");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    assembleGlobalBoundarySystem(std::forward< decltype(kernel) >(kernel),
                                 boundary,
                                 std::forward< decltype(fval_getter) >(fval_getter),
                                 *m_matrix,
                                 asSpan(rhs_view),
                                 getDofMap(),
                                 m_condensation_manager,
                                 field_inds_ctwrpr,
                                 assembly_options,
                                 time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::applyDirichletBCs()
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

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::assertState(State                expected,
                                                           std::string_view     err_msg,
                                                           std::source_location src_loc) const
{
    util::throwingAssert(m_state == expected, err_msg, src_loc);
}

namespace detail
{
consteval auto makeEmptyDirichletBCDef(const detail::ProblemDef_c auto& problem_def)
{
    return std::array< typename std::decay_t< decltype(problem_def) >::value_type, 0 >{};
}
} // namespace detail

template < CondensationPolicy        CP,
           detail::ProblemDef_c auto problem_def,
           detail::ProblemDef_c auto dirichlet_def = detail::makeEmptyDirichletBCDef(problem_def) >
auto makeAlgebraicSystem(const MpiComm&       comm,
                         const MeshPartition& mesh,
                         CondensationPolicyTag< CP >,
                         ConstexprValue< problem_def >   problemdef_ctwrpr,
                         ConstexprValue< dirichlet_def > dbcdef_ctwrpr = {})
    -> std::shared_ptr< AlgebraicSystem< detail::deduceNFields(problem_def), CP > >
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto max_dofs_per_node = detail::deduceNFields(problem_def);
    return AlgebraicSystem< max_dofs_per_node, CP >::makeAlgebraicSystem(comm, mesh, problemdef_ctwrpr, dbcdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP