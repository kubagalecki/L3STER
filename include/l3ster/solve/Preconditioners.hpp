#ifndef L3STER_SOLVE_PRECONDITIONERS_HPP
#define L3STER_SOLVE_PRECONDITIONERS_HPP

#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr::solvers
{
// General note: The following is a bit of a Frankenstein's monster between static CRTP- and a dynamic
// polymorphism-based interfaces. This is due to the fact that L3STER strongly prefers compile-time dispatch, while
// Trilinos is an extremely object-oriented library.

class PreconditionerBase
{
public:
    virtual void compute()        = 0;
    virtual ~PreconditionerBase() = default;
};

template < typename CRTP >
class PreconditionerInterface : public tpetra_operator_t, public PreconditionerBase
{
public:
    Teuchos::RCP< const tpetra_map_t > getDomainMap() const final
    {
        return static_cast< const CRTP* >(this)->getMapImpl();
    }
    Teuchos::RCP< const tpetra_map_t > getRangeMap() const final
    {
        return static_cast< const CRTP* >(this)->getMapImpl();
    }
    bool hasTransposeApply() const final { return static_cast< const CRTP* >(this)->hasTransposeImpl(); }
    void apply(const tpetra_multivector_t& X,
               tpetra_multivector_t&       Y,
               Teuchos::ETransp            trans,
               val_t                       alpha,
               val_t                       beta) const final
    {
        util::throwingAssert(X.getNumVectors() == Y.getNumVectors(), "X and Y must have the same number of columns");
        static_cast< const CRTP* >(this)->applyImpl(X, Y, trans, alpha, beta);
    }
    void compute() override { static_cast< CRTP* >(this)->computeImpl(); }

    virtual ~PreconditionerInterface() = default;
};

class RichardsonPreconditioner final : public PreconditionerInterface< RichardsonPreconditioner >
{
public:
    RichardsonPreconditioner(const Teuchos::RCP< const tpetra_operator_t >& A, val_t damp, int sweeps)
        : m_map{A->getRangeMap()}, m_damping_factor{damp}, m_sweeps{sweeps}
    {}

    void computeImpl() {}
    auto getMapImpl() const -> Teuchos::RCP< const tpetra_map_t > { return m_map; }
    bool hasTransposeImpl() const { return true; }
    void applyImpl(const tpetra_multivector_t& X,
                   tpetra_multivector_t&       Y,
                   Teuchos::ETransp, // Operator is symmetric
                   val_t alpha,
                   val_t beta) const
    {
        for (int i = 0; i != m_sweeps; ++i)
            Y.update(m_damping_factor * alpha, X, beta);
    }

private:
    Teuchos::RCP< const tpetra_map_t > m_map;
    double                             m_damping_factor;
    int                                m_sweeps;
};

class JacobiPreconditioner final : public PreconditionerInterface< JacobiPreconditioner >
{
public:
    JacobiPreconditioner(const Teuchos::RCP< const tpetra_operator_t >& A, val_t damping, val_t threshold, int sweeps)
        : m_source{A},
          m_diag_inv_damped{util::makeTeuchosRCP< tpetra_vector_t >(A->getDomainMap())},
          m_damping{damping},
          m_threshold{threshold},
          m_sweeps{sweeps}
    {
        util::throwingAssert(A->hasDiagonal(),
                             "Jacobi preconditioner must be initialized with a diagonal-aware operator");
    }

    inline void computeImpl();
    auto        getMapImpl() const -> Teuchos::RCP< const tpetra_map_t > { return m_diag_inv_damped->getMap(); }
    bool        hasTransposeImpl() const { return true; }
    void        applyImpl(const tpetra_multivector_t& X,
                          tpetra_multivector_t&       Y,
                          Teuchos::ETransp, // Operator is symmetric
                          val_t alpha,
                          val_t beta) const
    {
        for (int i = 0; i != m_sweeps; ++i)
            Y.elementWiseMultiply(alpha, *m_diag_inv_damped, X, beta);
    }

private:
    Teuchos::RCP< const tpetra_operator_t > m_source;
    Teuchos::RCP< tpetra_vector_t >         m_diag_inv_damped;
    val_t                                   m_damping, m_threshold;
    int                                     m_sweeps;
};

void JacobiPreconditioner::computeImpl()
{
    m_source->getLocalDiagCopy(*m_diag_inv_damped);
    const auto diag_view_2d    = m_diag_inv_damped->getLocalViewHost(Tpetra::Access::ReadWrite);
    const auto diag_view       = Kokkos::subview(diag_view_2d, Kokkos::ALL, 0);
    const auto diag_len        = diag_view.extent(0);
    const auto damping         = m_damping;   // Avoid implicit `this` capture
    const auto threshold       = m_threshold; // Avoid implicit `this` capture
    const auto diag_inv_kernel = KOKKOS_LAMBDA(const int i)
    {
        // Note: the following gets vectorized by the compiler, confirmed in Godbolt
        const auto  val       = diag_view(i);
        const val_t sign      = val < 0 ? -1. : 1.;
        const auto  scale     = sign * damping;
        const auto  val_abs   = std::fabs(val);
        const auto  val_fixed = std::max(val_abs, threshold);
        diag_view(i)          = scale / val_fixed;
    };
    Kokkos::parallel_for("Invert diagonal for Jacobi preconditioner", diag_len, diag_inv_kernel);
}
} // namespace lstr::solvers
#endif // L3STER_SOLVE_PRECONDITIONERS_HPP
