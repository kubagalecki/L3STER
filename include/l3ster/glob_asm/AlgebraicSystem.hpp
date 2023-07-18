#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP

#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
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
template < size_t max_dofs_per_node, CondensationPolicy CP >
class AlgebraicSystem
{
    template < el_o_t... orders, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def >
    AlgebraicSystem(const MpiComm&                          comm,
                    const mesh::MeshPartition< orders... >& mesh,
                    util::ConstexprValue< problem_def >,
                    util::ConstexprValue< dirichlet_def >);

public:
    template < el_o_t... orders, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def >
    static auto makeAlgebraicSystem(const MpiComm&                          comm,
                                    const mesh::MeshPartition< orders... >& mesh,
                                    util::ConstexprValue< problem_def >     problemdef_ctwrpr,
                                    util::ConstexprValue< dirichlet_def >   dbcdef_ctwrpr)
        -> std::shared_ptr< AlgebraicSystem >;

    [[nodiscard]] inline auto getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t >;
    [[nodiscard]] inline auto getRhs() const -> Teuchos::RCP< const tpetra_multivector_t >;
    [[nodiscard]] const auto& getDofMap() const { return m_node_dof_map; }

    [[nodiscard]] inline auto makeSolutionVector() const -> Teuchos::RCP< tpetra_femultivector_t >;
    template < el_o_t... orders >
    void updateSolution(const mesh::MeshPartition< orders... >&             mesh,
                        const Teuchos::RCP< const tpetra_femultivector_t >& solution,
                        IndexRange_c auto&&                                 sol_inds,
                        SolutionManager&                                    sol_man,
                        IndexRange_c auto&&                                 sol_man_inds);

    inline void beginAssembly();
    template < el_o_t... orders >
    inline void endAssembly(const mesh::MeshPartition< orders... >& mesh);

    template < ArrayOf_c< size_t > auto field_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{},
               el_o_t... orders >
    void assembleDomainProblem(auto&&                                               kernel,
                               const mesh::MeshPartition< orders... >&              mesh,
                               mesh::detail::DomainIdRange_c auto&&                 domain_ids,
                               const SolutionManager::FieldValueGetter< n_fields >& fval_getter       = {},
                               util::ConstexprValue< field_inds >                   field_inds_ctwrpr = {},
                               util::ConstexprValue< asm_opts >                     assembly_options  = {},
                               val_t                                                time              = 0.);
    template < ArrayOf_c< size_t > auto field_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t                   n_fields   = 0,
               AssemblyOptions          asm_opts   = AssemblyOptions{},
               el_o_t... orders >
    void assembleBoundaryProblem(auto&&                                               kernel,
                                 const mesh::BoundaryView< orders... >&               boundary,
                                 const SolutionManager::FieldValueGetter< n_fields >& fval_getter       = {},
                                 util::ConstexprValue< field_inds >                   field_inds_ctwrpr = {},
                                 util::ConstexprValue< asm_opts >                     assembly_options  = {},
                                 val_t                                                time              = 0.);

    inline void applyDirichletBCs();
    template < IndexRange_c auto dof_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t            n_fields = 0,
               el_o_t... orders >
    void setDirichletBCValues(auto&&                                               kernel,
                              const mesh::MeshPartition< orders... >&              mesh,
                              mesh::detail::DomainIdRange_c auto&&                 domain_ids,
                              util::ConstexprValue< dof_inds >                     dofinds_ctwrpr   = {},
                              const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = {},
                              val_t                                                time             = 0.);
    template < IndexRange_c auto dof_inds = util::makeIotaArray< size_t, max_dofs_per_node >(),
               size_t            n_fields = 0,
               el_o_t... orders >
    void setDirichletBCValues(auto&&                                               kernel,
                              const mesh::BoundaryView< orders... >&               mesh,
                              util::ConstexprValue< dof_inds >                     dofinds_ctwrpr   = {},
                              const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = 0,
                              val_t                                                time             = 0.);

    void solve(solvers::Solver_c auto& solver, const Teuchos::RCP< tpetra_femultivector_t >& solution) const;

    inline void describe(const MpiComm& comm, std::ostream& out = std::cout) const;

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

    Teuchos::RCP< tpetra_fecrsmatrix_t >             m_matrix;
    Teuchos::RCP< tpetra_femultivector_t >           m_rhs;
    Teuchos::RCP< const tpetra_fecrsgraph_t >        m_sparsity_graph;
    std::optional< const bcs::DirichletBCAlgebraic > m_dirichlet_bcs;
    Teuchos::RCP< tpetra_multivector_t >             m_dirichlet_values;
    dofs::NodeToLocalDofMap< max_dofs_per_node, 3 >  m_node_dof_map;
    detail::StaticCondensationManager< CP >          m_condensation_manager;
    State                                            m_state;

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
    template < el_o_t... orders, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def >
    static auto makeCacheKey(const mesh::MeshPartition< orders... >& mesh,
                             util::ConstexprValue< problem_def >,
                             util::ConstexprValue< dirichlet_def >) -> cache_key_t
    {
        return std::make_pair(util::type_id_value< util::ValuePack< problem_def, dirichlet_def > >,
                              mesh.computeTopoHash());
    }
};

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::solve(solvers::Solver_c auto&                       solver,
                                                     const Teuchos::RCP< tpetra_femultivector_t >& solution) const
{
    L3STER_PROFILE_FUNCTION;
    solution->switchActiveMultiVector();
    solver.solve(m_matrix, m_rhs, solution);
    solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    solution->switchActiveMultiVector();
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP >::updateSolution(
    const mesh::MeshPartition< orders... >&             mesh,
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
                                           util::asSpan(solution_view),
                                           std::forward< decltype(sol_inds) >(sol_inds),
                                           sol_man,
                                           std::forward< decltype(sol_man_inds) >(sol_man_inds));
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < IndexRange_c auto dof_inds, size_t n_fields, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP >::setDirichletBCValues(
    auto&&                                               kernel,
    const mesh::MeshPartition< orders... >&              mesh,
    mesh::detail::DomainIdRange_c auto&&                 domain_ids,
    util::ConstexprValue< dof_inds >                     dofinds_ctwrpr,
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
                         util::asSpan(vals_view),
                         time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < IndexRange_c auto dof_inds, size_t n_fields, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP >::setDirichletBCValues(
    auto&&                                               kernel,
    const mesh::BoundaryView< orders... >&               boundary_view,
    util::ConstexprValue< dof_inds >                     dofinds_ctwrpr,
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
                         util::asSpan(vals_view),
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
    return util::makeTeuchosRCP< tpetra_femultivector_t >(
        m_sparsity_graph->getRowMap(), m_sparsity_graph->getImporter(), 1u);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < el_o_t... orders, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def >
auto AlgebraicSystem< max_dofs_per_node, CP >::makeAlgebraicSystem(
    const MpiComm&                          comm,
    const mesh::MeshPartition< orders... >& mesh,
    util::ConstexprValue< problem_def >     problemdef_ctwrpr,
    util::ConstexprValue< dirichlet_def >   dbcdef_ctwrpr) -> std::shared_ptr< AlgebraicSystem >
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

template < el_o_t... orders, ProblemDef_c auto problem_def, CondensationPolicy CP >
AlgebraicSystem(const MpiComm&,
                const mesh::MeshPartition< orders... >&,
                util::ConstexprValue< problem_def >,
                CondensationPolicyTag< CP >) -> AlgebraicSystem< detail::deduceNFields(problem_def), CP >;
template < el_o_t... orders, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def, CondensationPolicy CP >
AlgebraicSystem(const MpiComm&,
                const mesh::MeshPartition< orders... >&,
                util::ConstexprValue< problem_def >,
                util::ConstexprValue< dirichlet_def >,
                CondensationPolicyTag< CP >) -> AlgebraicSystem< detail::deduceNFields(problem_def), CP >;

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < el_o_t... orders, ProblemDef_c auto problem_def, ProblemDef_c auto dirichlet_def >
AlgebraicSystem< max_dofs_per_node, CP >::AlgebraicSystem(const MpiComm&                          comm,
                                                          const mesh::MeshPartition< orders... >& mesh,
                                                          util::ConstexprValue< problem_def >     problemdef_ctwrpr,
                                                          util::ConstexprValue< dirichlet_def >   dbcdef_ctwrpr)
    : m_state{State::OpenForAssembly}
{
    const auto cond_map            = dofs::makeCondensationMap< CP >(comm, mesh, problemdef_ctwrpr);
    const auto dof_intervals       = computeDofIntervals(comm, mesh, cond_map, problemdef_ctwrpr);
    const auto node_global_dof_map = dofs::NodeToGlobalDofMap{dof_intervals, cond_map};
    m_sparsity_graph = detail::makeSparsityGraph(comm, mesh, node_global_dof_map, cond_map, problemdef_ctwrpr);

    L3STER_PROFILE_REGION_BEGIN("Create Tpetra objects");
    m_matrix = util::makeTeuchosRCP< tpetra_fecrsmatrix_t >(m_sparsity_graph);
    m_rhs    = makeSolutionVector();
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    L3STER_PROFILE_REGION_END("Create Tpetra objects");

    m_node_dof_map = dofs::NodeToLocalDofMap{
        cond_map, node_global_dof_map, *m_matrix->getRowMap(), *m_matrix->getColMap(), *m_rhs->getMap()};
    m_condensation_manager = detail::StaticCondensationManager< CP >{mesh, m_node_dof_map, problemdef_ctwrpr};
    m_condensation_manager.beginAssembly();

    if constexpr (dirichlet_def.size() != 0)
    {
        L3STER_PROFILE_REGION_BEGIN("Dirichlet BCs");
        auto [owned_bcdofs, shared_bcdofs] = bcs::getDirichletDofs(
            mesh, m_sparsity_graph, node_global_dof_map, cond_map, problemdef_ctwrpr, dbcdef_ctwrpr);
        m_dirichlet_bcs.emplace(m_sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
        L3STER_PROFILE_REGION_END("Dirichlet BCs");
    }
    if (m_dirichlet_bcs.has_value())
        m_dirichlet_values = util::makeTeuchosRCP< tpetra_multivector_t >(m_sparsity_graph->getRowMap(), 1u);
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
template < el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP >::endAssembly(const mesh::MeshPartition< orders... >& mesh)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`endAssembly()` was called more than once");

    L3STER_PROFILE_REGION_BEGIN("Static condensation");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    m_condensation_manager.endAssembly(mesh, m_node_dof_map, *m_matrix, util::asSpan(rhs_view));
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
template < ArrayOf_c< size_t > auto field_inds, size_t n_fields, AssemblyOptions asm_opts, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP >::assembleDomainProblem(
    auto&&                                               kernel,
    const mesh::MeshPartition< orders... >&              mesh,
    mesh::detail::DomainIdRange_c auto&&                 domain_ids,
    const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
    util::ConstexprValue< field_inds >                   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >                     assembly_options,
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
                         util::asSpan(rhs_view),
                         getDofMap(),
                         m_condensation_manager,
                         field_inds_ctwrpr,
                         assembly_options,
                         time);
}

template < size_t max_dofs_per_node, CondensationPolicy CP >
template < ArrayOf_c< size_t > auto field_inds, size_t n_fields, AssemblyOptions asm_opts, el_o_t... orders >
void AlgebraicSystem< max_dofs_per_node, CP >::assembleBoundaryProblem(
    auto&&                                               kernel,
    const mesh::BoundaryView< orders... >&               boundary,
    const SolutionManager::FieldValueGetter< n_fields >& fval_getter,
    util::ConstexprValue< field_inds >                   field_inds_ctwrpr,
    util::ConstexprValue< asm_opts >                     assembly_options,
    val_t                                                time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(State::OpenForAssembly, "`assembleBoundaryProblem()` was called before `beginAssembly()`");
    const auto rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    assembleGlobalBoundarySystem(std::forward< decltype(kernel) >(kernel),
                                 boundary,
                                 std::forward< decltype(fval_getter) >(fval_getter),
                                 *m_matrix,
                                 util::asSpan(rhs_view),
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

template < size_t max_dofs_per_node, CondensationPolicy CP >
void AlgebraicSystem< max_dofs_per_node, CP >::describe(const MpiComm& comm, std::ostream& out) const
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

namespace detail
{
consteval auto makeEmptyDirichletBCDef(const ProblemDef_c auto& problem_def)
{
    return std::array< typename std::decay_t< decltype(problem_def) >::value_type, 0 >{};
}
} // namespace detail

template < el_o_t... orders,
           CondensationPolicy CP,
           ProblemDef_c auto  problem_def,
           ProblemDef_c auto  dirichlet_def = detail::makeEmptyDirichletBCDef(problem_def) >
auto makeAlgebraicSystem(const MpiComm&                          comm,
                         const mesh::MeshPartition< orders... >& mesh,
                         CondensationPolicyTag< CP >,
                         util::ConstexprValue< problem_def >   problemdef_ctwrpr,
                         util::ConstexprValue< dirichlet_def > dbcdef_ctwrpr = {})
    -> std::shared_ptr< AlgebraicSystem< detail::deduceNFields(problem_def), CP > >
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto max_dofs_per_node = detail::deduceNFields(problem_def);
    return AlgebraicSystem< max_dofs_per_node, CP >::makeAlgebraicSystem(comm, mesh, problemdef_ctwrpr, dbcdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
