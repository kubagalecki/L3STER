#ifndef L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
#define L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP

#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"

#include <typeinfo>

namespace lstr
{
template < size_t n_fields >
class AlgebraicSystemManager
{
public:
    using map_t           = Tpetra::Map< local_dof_t, global_dof_t >;
    using fematrix_t      = Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t >;
    using fevector_t      = Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >;
    using mltvector_t     = Tpetra::MultiVector< val_t, local_dof_t, global_dof_t >;
    using vector_t        = Tpetra::Vector< val_t, local_dof_t, global_dof_t >;
    using dof_map_local_t = NodeToLocalDofMap< n_fields >;
    using bc_t            = DirichletBCAlgebraic;

    template < detail::ProblemDef_c auto problem_def >
    AlgebraicSystemManager(const MpiComm& comm, const MeshPartition& mesh, ConstexprValue< problem_def >);
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    AlgebraicSystemManager(const MpiComm&       comm,
                           const MeshPartition& mesh,
                           ConstexprValue< problem_def >,
                           ConstexprValue< dirichlet_def >);

    const auto& getMatrix() const { return m_cache_ptr->matrix; }
    const auto& getRhs() const { return m_cache_ptr->rhs; }
    const auto& getRowMap() const { return m_cache_ptr->row_map; }
    const auto& getColMap() const { return m_cache_ptr->col_map; }
    const auto& getRhsMap() const { return m_cache_ptr->rhs_map; }

    [[nodiscard]] inline Teuchos::RCP< mltvector_t > makeSolutionMultiVector(size_t n_cols = 1) const;

    inline void beginAssembly();
    inline void endAssembly();
    inline void beginModify();
    inline void endModify();
    inline void setToZero();

    template < BasisTypes               BT,
               QuadratureTypes          QT,
               q_o_t                    QO,
               ArrayOf_c< size_t > auto field_inds,
               typename Kernel,
               detail::FieldValGetter_c FvalGetter,
               detail::DomainIdRange_c  R >
    void assembleDomainProblem(
        Kernel&& kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& fval_getter, val_t time = 0.);
    template < BasisTypes               BT,
               QuadratureTypes          QT,
               q_o_t                    QO,
               ArrayOf_c< size_t > auto field_inds,
               typename Kernel,
               detail::FieldValGetter_c FvalGetter >
    void
    assembleBoundaryProblem(Kernel&& kernel, const BoundaryView& boundary, FvalGetter&& fval_getter, val_t time = 0.);
    inline void applyDirichletBCs(const vector_t& bc_vals);

private:
    enum class State
    {
        OpenForAssembly,
        OpenForModify,
        Closed
    };

    // Caching mechanism to enable the reuse of the system allocation. For example, an adjoint problem will have the
    // same structure as the primal problem. We can therefore reuse the assembly data structures from the primal.
    // Note that this implies that assembly should soon be followed by a solve, otherwise data may be overwritten.
    //
    // The cache has weak ownership semantics, since we need all instances of Trilinos objects to be destroyed before
    // MPI and Kokkos are finalized.
    struct CacheTagBase
    {
        CacheTagBase()                                   = default;
        CacheTagBase(const CacheTagBase&)                = default;
        CacheTagBase& operator=(const CacheTagBase&)     = default;
        CacheTagBase(CacheTagBase&&) noexcept            = default;
        CacheTagBase& operator=(CacheTagBase&&) noexcept = default;
        virtual ~CacheTagBase()                          = default;
    };
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dbc_def >
    struct CacheTag : CacheTagBase
    {};
    struct CacheKey
    {
        std::unique_ptr< CacheTagBase > tag;
        size_t                          mesh_hash; // Assume no collisions (there's likely only 1 mesh anyway)

        bool operator<(const CacheKey& other) const
        {
            return std::array{mesh_hash, typeid(*tag).hash_code()} <
                   std::array{other.mesh_hash, typeid(*other.tag).hash_code()};
        }
    };
    struct CacheEntry
    {
        State                       state;
        Teuchos::RCP< fematrix_t >  matrix;
        Teuchos::RCP< fevector_t >  rhs;
        std::optional< const bc_t > dirichlet_bcs; // May be null
        const dof_map_local_t       row_map, col_map, rhs_map;
        Teuchos::RCP< const map_t > owned_map;
    };
    static inline std::map< CacheKey, std::weak_ptr< CacheEntry > > cache{};

    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    static std::shared_ptr< CacheEntry > initOrRetrieveFromCache(const MpiComm&       comm,
                                                                 const MeshPartition& mesh,
                                                                 ConstexprValue< problem_def >,
                                                                 ConstexprValue< dirichlet_def >);
    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    static CacheEntry makeCacheEntry(const MpiComm&       comm,
                                     const MeshPartition& mesh,
                                     ConstexprValue< problem_def >,
                                     ConstexprValue< dirichlet_def >);

    template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
    void initFromCache(const MpiComm&       comm,
                       const MeshPartition& mesh,
                       ConstexprValue< problem_def >,
                       ConstexprValue< dirichlet_def >);

    inline void openRhs();
    inline void closeRhs();

    // We need to hold strong ownership of the cache entry (the cache holds weak ownership)
    std::shared_ptr< CacheEntry > m_cache_ptr;
    Teuchos::ArrayRCP< val_t >    m_rhs_alloc; // Host view of the rhs
    std::span< val_t >            m_rhs_view;  // We need a thread safe view, which Teuchos::ArrayRCP is not
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
AlgebraicSystemManager< n_fields >::AlgebraicSystemManager(const MpiComm&                comm,
                                                           const MeshPartition&          mesh,
                                                           ConstexprValue< problem_def > problemdef_ctwrpr)
{
    using empty_bc = std::array< typename decltype(problem_def)::value_type, 0 >;
    initFromCache(comm, mesh, problemdef_ctwrpr, ConstexprValue< empty_bc{} >{});
}
template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
AlgebraicSystemManager< n_fields >::AlgebraicSystemManager(const MpiComm&                  comm,
                                                           const MeshPartition&            mesh,
                                                           ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                           ConstexprValue< dirichlet_def > bcdef_ctwrpr)
{
    initFromCache(comm, mesh, problemdef_ctwrpr, bcdef_ctwrpr);
}

template < size_t n_fields >
Teuchos::RCP< Tpetra::MultiVector< val_t, local_dof_t, global_dof_t > >
AlgebraicSystemManager< n_fields >::makeSolutionMultiVector(size_t n_cols) const
{
    return makeTeuchosRCP< mltvector_t >(m_cache_ptr->owned_map, n_cols, false);
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::beginAssembly()
{
    switch (m_cache_ptr->state)
    {
    case State::OpenForAssembly:
        return;
    case State::OpenForModify:
        throw std::runtime_error{
            "Initiation of assembly was attempted while the algebraic system was in the \"open for modification\" "
            "state. Finalize the modification first before calling \"beginAssembly\"."};
    case State::Closed:
        m_cache_ptr->matrix->beginAssembly();
        m_cache_ptr->rhs->beginAssembly();
        openRhs();
        m_cache_ptr->state = State::OpenForAssembly;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::endAssembly()
{
    switch (m_cache_ptr->state)
    {
    case State::OpenForAssembly:
        m_cache_ptr->matrix->endAssembly();
        closeRhs();
        m_cache_ptr->rhs->endAssembly();
        m_cache_ptr->state = State::Closed;
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
    switch (m_cache_ptr->state)
    {
    case State::OpenForAssembly:
        throw std::runtime_error{
            "Initiation of modification was attempted while the algebraic system was in the \"open for assembly\" "
            "state. Finalize the asembly first before calling \"beginModify\"."};
    case State::OpenForModify:
        return;
    case State::Closed:
        m_cache_ptr->matrix->beginModify();
        m_cache_ptr->rhs->beginModify();
        openRhs();
        m_cache_ptr->state = State::OpenForModify;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::endModify()
{
    switch (m_cache_ptr->state)
    {
    case State::OpenForAssembly:
        throw std::runtime_error{
            "Finilization of modification was attempted while the algebraic system was in the \"open for assembly\" "
            "state."};
    case State::OpenForModify:
        m_cache_ptr->matrix->endModify();
        closeRhs();
        m_cache_ptr->rhs->endModify();
        m_cache_ptr->state = State::Closed;
        break;
    case State::Closed:
        return;
    }
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::setToZero()
{
    if (m_cache_ptr->state == State::Closed)
        throw std::runtime_error{"The system may only be zeroed if it is in a non-closed state."};
    m_cache_ptr->matrix->setAllToScalar(0.);
    m_cache_ptr->rhs->putScalar(0.);
}

template < size_t n_fields >
template < BasisTypes               BT,
           QuadratureTypes          QT,
           q_o_t                    QO,
           ArrayOf_c< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter,
           detail::DomainIdRange_c  R >
void AlgebraicSystemManager< n_fields >::assembleDomainProblem(
    Kernel&& kernel, const MeshPartition& mesh, R&& domain_ids, FvalGetter&& fval_getter, val_t time)
{
    if (m_cache_ptr->state == State::Closed)
        throw std::runtime_error{"Assemble was called while the algebraic system was in a closed state."};
    assembleGlobalSystem< BT, QT, QO, field_inds >(std::forward< Kernel >(kernel),
                                                   mesh,
                                                   std::forward< R >(domain_ids),
                                                   std::forward< FvalGetter >(fval_getter),
                                                   *m_cache_ptr->matrix,
                                                   m_rhs_view,
                                                   getRowMap(),
                                                   getColMap(),
                                                   getRhsMap(),
                                                   time);
}

template < size_t n_fields >
template < BasisTypes               BT,
           QuadratureTypes          QT,
           q_o_t                    QO,
           ArrayOf_c< size_t > auto field_inds,
           typename Kernel,
           detail::FieldValGetter_c FvalGetter >
void AlgebraicSystemManager< n_fields >::assembleBoundaryProblem(Kernel&&            kernel,
                                                                 const BoundaryView& boundary,
                                                                 FvalGetter&&        fval_getter,
                                                                 val_t               time)
{
    if (m_cache_ptr->state == State::Closed)
        throw std::runtime_error{"Assemble was called while the algebraic system was in a closed state."};
    assembleGlobalBoundarySystem< BT, QT, QO, field_inds >(std::forward< Kernel >(kernel),
                                                           boundary,
                                                           std::forward< FvalGetter >(fval_getter),
                                                           *m_cache_ptr->matrix,
                                                           m_rhs_view,
                                                           getRowMap(),
                                                           getColMap(),
                                                           getRhsMap(),
                                                           time);
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::applyDirichletBCs(
    const Tpetra::Vector< val_t, local_dof_t, global_dof_t >& bc_vals)
{
    if (not m_cache_ptr->dirichlet_bcs)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but no Dirichlet BCs were defined."};
    if (m_cache_ptr->state != State::OpenForModify)
        throw std::runtime_error{"Application of Dirichlet BCs was attempted, but the system was not in the \"open for "
                                 "modification\" state."};
    m_cache_ptr->dirichlet_bcs->apply(bc_vals, *m_cache_ptr->matrix, *m_cache_ptr->rhs->getVectorNonConst(0));
}

template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::openRhs()
{
    m_cache_ptr->rhs->sync_host();
    m_cache_ptr->rhs->modify_host();
    m_rhs_alloc = m_cache_ptr->rhs->getDataNonConst(0);
    m_rhs_view  = m_rhs_alloc;
}
template < size_t n_fields >
void AlgebraicSystemManager< n_fields >::closeRhs()
{
    m_rhs_view  = {};
    m_rhs_alloc = {};
    m_cache_ptr->rhs->sync_device();
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
void AlgebraicSystemManager< n_fields >::initFromCache(const MpiComm&                  comm,
                                                       const MeshPartition&            mesh,
                                                       ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                       ConstexprValue< dirichlet_def > bcdef_ctwrpr)
{
    m_cache_ptr = initOrRetrieveFromCache(comm, mesh, problemdef_ctwrpr, bcdef_ctwrpr);
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
std::shared_ptr< typename AlgebraicSystemManager< n_fields >::CacheEntry >
AlgebraicSystemManager< n_fields >::initOrRetrieveFromCache(const MpiComm&                  comm,
                                                            const MeshPartition&            mesh,
                                                            ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                            ConstexprValue< dirichlet_def > dbcdef_ctwrpr)
{
    auto       tag             = std::unique_ptr< CacheTagBase >(new CacheTag< problem_def, dirichlet_def >);
    const auto mesh_hash       = mesh.computeTopoHash();
    auto       cache_key       = CacheKey{std::move(tag), mesh_hash};
    const auto lookup_result   = cache.find(cache_key);
    const char local_cache_hit = lookup_result != end(cache) and not lookup_result->second.expired();
    char       global_cache_hit{};
    comm.allReduce(&local_cache_hit, &global_cache_hit, 1, MPI_LAND);

    if (global_cache_hit)
        return lookup_result->second.lock();

    auto cache_entry = std::make_shared< CacheEntry >(makeCacheEntry(comm, mesh, problemdef_ctwrpr, dbcdef_ctwrpr));
    cache.insert_or_assign(std::move(cache_key), cache_entry);
    return cache_entry;
}

template < size_t n_fields >
template < detail::ProblemDef_c auto problem_def, detail::ProblemDef_c auto dirichlet_def >
typename AlgebraicSystemManager< n_fields >::CacheEntry
AlgebraicSystemManager< n_fields >::makeCacheEntry(const MpiComm&                  comm,
                                                   const MeshPartition&            mesh,
                                                   ConstexprValue< problem_def >   problemdef_ctwrpr,
                                                   ConstexprValue< dirichlet_def > bcdef_ctwrpr)
{
    const auto dof_intervals       = computeDofIntervals(mesh, problemdef_ctwrpr, comm);
    const auto node_global_dof_map = NodeToGlobalDofMap< n_fields >{mesh, dof_intervals};

    const auto sparsity_graph = detail::makeSparsityGraph(mesh, problemdef_ctwrpr, dof_intervals, comm);
    auto       matrix         = makeTeuchosRCP< fematrix_t >(sparsity_graph);
    auto       rhs     = makeTeuchosRCP< fevector_t >(sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1u);
    auto       row_map = NodeToLocalDofMap< n_fields >{mesh, node_global_dof_map, *matrix->getRowMap()};
    auto       col_map = NodeToLocalDofMap< n_fields >{mesh, node_global_dof_map, *matrix->getColMap()};
    auto       rhs_map = NodeToLocalDofMap< n_fields >{mesh, node_global_dof_map, *rhs->getMap()};

    matrix->beginAssembly();
    rhs->beginAssembly();

    std::optional< bc_t > dbcs;
    if constexpr (dirichlet_def.size() != 0)
    {
        auto [owned_bcdofs, shared_bcdofs] =
            detail::getDirichletDofs(mesh, sparsity_graph, node_global_dof_map, problemdef_ctwrpr, bcdef_ctwrpr);
        dbcs.emplace(sparsity_graph, std::move(owned_bcdofs), std::move(shared_bcdofs));
    }

    return CacheEntry{.state         = State::OpenForAssembly,
                      .matrix        = std::move(matrix),
                      .rhs           = std::move(rhs),
                      .dirichlet_bcs = std::move(dbcs),
                      .row_map       = std::move(row_map),
                      .col_map       = std::move(col_map),
                      .rhs_map       = std::move(rhs_map),
                      .owned_map     = sparsity_graph->getRowMap()};
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ALGEBRAICSYSTEMMANAGER_HPP
