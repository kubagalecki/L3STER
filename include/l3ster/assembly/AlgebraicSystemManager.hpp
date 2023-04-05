#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP

#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/assembly/StaticCondensationManager.hpp"
#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/TypeID.hpp"
#include "l3ster/util/WeakCache.hpp"

namespace lstr
{
template < size_t n_fields, CondensationPolicy CP >
class AlgebraicSystemManager
{
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    AlgebraicSystemManager(const MpiComm&       comm,
                           const MeshPartition& mesh,
                           ConstexprValue< problem_def >,
                           ConstexprValue< dirichlet_def >);

public:
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    static auto makeAlgebraicSystemManager(const MpiComm&                  comm,
                                           const MeshPartition&            mesh,
                                           ConstexprValue< problem_def >   problemdef_ctwrpr,
                                           ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
        -> std::shared_ptr< AlgebraicSystemManager >;

    [[nodiscard]] auto        getMatrix() const -> Teuchos::RCP< const tpetra_crsmatrix_t > { return m_matrix; }
    [[nodiscard]] auto        getRhs() const -> Teuchos::RCP< const tpetra_multivector_t > { return m_rhs; }
    [[nodiscard]] const auto& getDofMap() const { return m_node_dof_map; }
    [[nodiscard]] auto        getSolutionVector() const { return m_owned_values->getVector(0); }
    [[nodiscard]] auto        getSolutionVector() { return m_owned_values->getVectorNonConst(0); }
    [[nodiscard]] auto        getDirichletBCValueVector() const { return m_owned_values->getVector(1); }
    [[nodiscard]] auto        getDirichletBCValueVector() { return m_owned_values->getVectorNonConst(1); }

    inline void beginAssembly();
    inline void endAssembly();
    inline void beginModify();
    inline void endModify();

    template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, ArrayOf_c< size_t > auto field_inds >
    void assembleDomainProblem(auto&&                          kernel,
                               const MeshPartition&            mesh,
                               detail::DomainIdRange_c auto&&  domain_ids,
                               detail::FieldValGetter_c auto&& fval_getter,
                               val_t                           time = 0.);
    template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, ArrayOf_c< size_t > auto field_inds >
    void        assembleBoundaryProblem(auto&&                          kernel,
                                        const BoundaryView&             boundary,
                                        detail::FieldValGetter_c auto&& fval_getter,
                                        val_t                           time = 0.);
    inline void applyDirichletBCs();

private:
    inline void setToZero();

    enum struct State
    {
        OpenForAssembly = 0b001,
        OpenForModify   = 0b010,
        Closed          = 0b100
    };
    inline void assertState(State expected, const char* err_msg);

    using rhs_view_t = decltype(Kokkos::subview(
        std::declval< const tpetra_multivector_t::dual_view_type::t_host& >(), Kokkos::ALL, 0));

    State                                       m_state;
    Teuchos::RCP< tpetra_fecrsmatrix_t >        m_matrix;
    Teuchos::RCP< tpetra_multivector_t >        m_owned_values;  // solution + Dirichlet BC vals (if defined)
    Teuchos::RCP< tpetra_femultivector_t >      m_rhs;
    rhs_view_t                                  m_rhs_view;      // Thread-safe view
    std::optional< const DirichletBCAlgebraic > m_dirichlet_bcs; // May be null
    NodeToLocalDofMap< n_fields, 3 >            m_node_dof_map;
    detail::StaticCondensationManager< CP >     m_condensation_manager;

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
    using cache_t          = WeakCache< cache_key_t, AlgebraicSystemManager, cache_key_hash_t >;
    static inline cache_t cache{};
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    static auto makeCacheKey(const MeshPartition& mesh, ConstexprValue< problem_def >, ConstexprValue< dirichlet_def >)
        -> cache_key_t
    {
        return std::make_pair(type_id_value< ValuePack< problem_def, dirichlet_def > >, mesh.computeTopoHash());
    }
};

template < size_t n_fields, CondensationPolicy CP >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
auto AlgebraicSystemManager< n_fields, CP >::makeAlgebraicSystemManager(const MpiComm&                comm,
                                                                        const MeshPartition&          mesh,
                                                                        ConstexprValue< problem_def > problemdef_ctwrpr,
                                                                        ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
    -> std::shared_ptr< AlgebraicSystemManager >
{
    auto       cache_key       = makeCacheKey(mesh, problemdef_ctwrpr, dbcdef_ctwrpr);
    const char local_cache_hit = cache.contains(cache_key);
    char       global_cache_hit{};
    comm.allReduce(std::views::single(local_cache_hit), &global_cache_hit, MPI_LAND);
    if (global_cache_hit)
        return cache.get(cache_key);
    else
        return cache.emplace(cache_key, AlgebraicSystemManager{comm, mesh, problemdef_ctwrpr, dbcdef_ctwrpr});
}

template < detail::ProblemDef_c auto problem_def, CondensationPolicy CP >
AlgebraicSystemManager(const MpiComm&, const MeshPartition&, ConstexprValue< problem_def >, CondensationPolicyTag< CP >)
    -> AlgebraicSystemManager< detail::deduceNFields(problem_def), CP >;
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def, CondensationPolicy CP >
AlgebraicSystemManager(const MpiComm&,
                       const MeshPartition&,
                       ConstexprValue< problem_def >,
                       ConstexprValue< dirichlet_def >,
                       CondensationPolicyTag< CP >) -> AlgebraicSystemManager< detail::deduceNFields(problem_def), CP >;

template < size_t n_fields, CondensationPolicy CP >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
AlgebraicSystemManager< n_fields, CP >::AlgebraicSystemManager(const MpiComm&                  comm,
                                                               const MeshPartition&            mesh,
                                                               ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                               ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
    : m_state{State::OpenForAssembly}
{
    const auto cond_map            = detail::makeCondensationMap< CP >(comm, mesh, problemdef_ctwrpr);
    const auto dof_intervals       = computeDofIntervals(comm, mesh, cond_map, problemdef_ctwrpr);
    const auto node_global_dof_map = NodeToGlobalDofMap{dof_intervals, cond_map};
    const auto sparsity_graph = detail::makeSparsityGraph(comm, mesh, node_global_dof_map, cond_map, problemdef_ctwrpr);

    L3STER_PROFILE_REGION_BEGIN("Create Tpetra objects");
    m_matrix = makeTeuchosRCP< tpetra_fecrsmatrix_t >(sparsity_graph);
    m_rhs    = makeTeuchosRCP< tpetra_femultivector_t >(sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1u);
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    m_rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    L3STER_PROFILE_REGION_END("Create Tpetra objects");

    m_node_dof_map = NodeToLocalDofMap{
        cond_map, node_global_dof_map, *m_matrix->getRowMap(), *m_matrix->getColMap(), *m_rhs->getMap()};

    if constexpr (dirichlet_def.size() != 0)
    {
        L3STER_PROFILE_REGION_BEGIN("Dirichlet BCs");
        auto [owned_bcdofs, shared_bcdofs] =
            detail::getDirichletDofs(mesh, sparsity_graph, node_global_dof_map, problemdef_ctwrpr, dbcdef_ctwrpr);
        m_dirichlet_bcs.emplace(sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
        L3STER_PROFILE_REGION_END("Dirichlet BCs");
    }
    m_owned_values =
        makeTeuchosRCP< tpetra_multivector_t >(sparsity_graph->getRowMap(), m_dirichlet_bcs.has_value() ? 2u : 1u);
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::beginAssembly()
{
    L3STER_PROFILE_FUNCTION;
    assertState(~State::OpenForModify,
                "Initiation of assembly was attempted while the algebraic system was in the \"open for modification\" "
                "state. Finalize the modification first before calling \"beginAssembly\".");
    if (m_state == State::OpenForAssembly)
        return;

    m_state = State::OpenForAssembly;
    m_matrix->beginAssembly();
    m_rhs->beginAssembly();
    m_rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::OverwriteAll), Kokkos::ALL, 0);
    setToZero();
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::endAssembly()
{
    L3STER_PROFILE_FUNCTION;
    assertState(~State::OpenForModify,
                "Finilization of assembly was attempted while the algebraic system was in the \"open for "
                "modification\" state.");
    if (m_state == State::Closed)
        return;

    m_rhs_view = {};
    L3STER_PROFILE_REGION_BEGIN("RHS");
    m_rhs->endAssembly();
    L3STER_PROFILE_REGION_END("RHS");
    L3STER_PROFILE_REGION_BEGIN("Matrix");
    m_matrix->endAssembly();
    L3STER_PROFILE_REGION_END("Matrix");
    m_state = State::Closed;
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::beginModify()
{
    L3STER_PROFILE_FUNCTION;
    assertState(~State::OpenForAssembly,
                "Initiation of modification was attempted while the algebraic system was in the \"open for assembly\" "
                "state. Finalize the asembly first before calling \"beginModify\".");
    if (m_state == State::OpenForModify)
        return;

    m_matrix->beginModify();
    m_rhs->beginModify();
    m_rhs_view = Kokkos::subview(m_rhs->getLocalViewHost(Tpetra::Access::ReadWrite), Kokkos::ALL, 0);
    m_state    = State::OpenForModify;
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::endModify()
{
    L3STER_PROFILE_FUNCTION;
    assertState(~State::OpenForAssembly,
                "Finilization of modification was attempted while the algebraic system was in the \"open for "
                "assembly\" state.");
    if (m_state == State::Closed)
        return;

    m_rhs_view = {};
    m_rhs->endModify();
    m_matrix->endModify();
    m_state = State::Closed;
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::setToZero()
{
    m_matrix->setAllToScalar(0.);
    m_rhs->putScalar(0.);
}

template < size_t n_fields, CondensationPolicy CP >
template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, ArrayOf_c< size_t > auto field_inds >
void AlgebraicSystemManager< n_fields, CP >::assembleDomainProblem(auto&&                          kernel,
                                                                   const MeshPartition&            mesh,
                                                                   detail::DomainIdRange_c auto&&  domain_ids,
                                                                   detail::FieldValGetter_c auto&& fval_getter,
                                                                   val_t                           time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(~State::Closed, "Assemble was called while the algebraic system was in a closed state.");
    assembleGlobalSystem< BT, QT, QO, field_inds >(std::forward< decltype(kernel) >(kernel),
                                                   mesh,
                                                   std::forward< decltype(domain_ids) >(domain_ids),
                                                   std::forward< decltype(fval_getter) >(fval_getter),
                                                   *m_matrix,
                                                   asSpan(m_rhs_view),
                                                   getDofMap(),
                                                   time);
}

template < size_t n_fields, CondensationPolicy CP >
template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, ArrayOf_c< size_t > auto field_inds >
void AlgebraicSystemManager< n_fields, CP >::assembleBoundaryProblem(auto&&                          kernel,
                                                                     const BoundaryView&             boundary,
                                                                     detail::FieldValGetter_c auto&& fval_getter,
                                                                     val_t                           time)
{
    L3STER_PROFILE_FUNCTION;
    assertState(~State::Closed, "Assemble was called while the algebraic system was in a closed state.");
    assembleGlobalBoundarySystem< BT, QT, QO, field_inds >(std::forward< decltype(kernel) >(kernel),
                                                           boundary,
                                                           std::forward< decltype(fval_getter) >(fval_getter),
                                                           *m_matrix,
                                                           asSpan(m_rhs_view),
                                                           getDofMap(),
                                                           time);
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::applyDirichletBCs()
{
    L3STER_PROFILE_FUNCTION;
    if (not m_dirichlet_bcs)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but no Dirichlet BCs were defined."};
    assertState(
        State::OpenForModify,
        "Application of Dirichlet BCs was attempted, but the system was not in the \"open for modification\" state.");
    m_dirichlet_bcs->apply(*std::as_const(*this).getDirichletBCValueVector(), *m_matrix, *m_rhs->getVectorNonConst(0));
}

template < size_t n_fields, CondensationPolicy CP >
void AlgebraicSystemManager< n_fields, CP >::assertState(State expected, const char* err_msg)
{
    if (not(m_state & expected))
        throw std::runtime_error{err_msg};
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
auto makeAlgebraicSystemManager(const MpiComm&       comm,
                                const MeshPartition& mesh,
                                CondensationPolicyTag< CP >,
                                ConstexprValue< problem_def >   problemdef_ctwrpr,
                                ConstexprValue< dirichlet_def > dbcdef_ctwrpr = {})
    -> std::shared_ptr< AlgebraicSystemManager< detail::deduceNFields(problem_def), CP > >
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto n_fields = detail::deduceNFields(problem_def);
    return AlgebraicSystemManager< n_fields, CP >::makeAlgebraicSystemManager(
        comm, mesh, problemdef_ctwrpr, dbcdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
