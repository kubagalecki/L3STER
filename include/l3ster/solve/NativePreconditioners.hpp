#ifndef L3STER_SOLVE_PRECONDITIONERS_HPP
#define L3STER_SOLVE_PRECONDITIONERS_HPP

#include "l3ster/solve/PreconditionerInterface.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

namespace lstr::solvers
{
class NativeRichardsonImpl final : public tpetra_operator_t
{
public:
    NativeRichardsonImpl(const Teuchos::RCP< const tpetra_operator_t >& A, val_t damp, int sweeps)
        : m_map{A->getRangeMap()}, m_damping_factor{damp}, m_sweeps{sweeps}
    {}

    Teuchos::RCP< const tpetra_map_t > getDomainMap() const override { return m_map; }
    Teuchos::RCP< const tpetra_map_t > getRangeMap() const override { return m_map; }
    bool                               hasTransposeApply() const override { return true; }
    void                               apply(const tpetra_multivector_t& X,
                                             tpetra_multivector_t&       Y,
                                             Teuchos::ETransp, // Operator is symmetric
                                             val_t alpha,
                                             val_t beta) const override
    {
        for (int i = 0; i != m_sweeps; ++i)
            Y.update(m_damping_factor * alpha, X, beta);
    }

private:
    Teuchos::RCP< const tpetra_map_t > m_map;
    double                             m_damping_factor;
    int                                m_sweeps;
};

class NativeJacobiImpl final : public tpetra_operator_t
{
public:
    NativeJacobiImpl(const Teuchos::RCP< const tpetra_operator_t >& A, val_t damping, val_t threshold, int sweeps)
        : m_diag_inv_damped{util::makeTeuchosRCP< tpetra_vector_t >(A->getDomainMap())},
          m_damping{damping},
          m_threshold{threshold},
          m_sweeps{sweeps}
    {
        init(A);
    }

    Teuchos::RCP< const tpetra_map_t > getDomainMap() const override { return m_diag_inv_damped->getMap(); }
    Teuchos::RCP< const tpetra_map_t > getRangeMap() const override { return m_diag_inv_damped->getMap(); }
    bool                               hasTransposeApply() const override { return true; }
    void                               apply(const tpetra_multivector_t& X,
                                             tpetra_multivector_t&       Y,
                                             Teuchos::ETransp, // Operator is symmetric
                                             val_t alpha,
                                             val_t beta) const override
    {
        L3STER_PROFILE_REGION_BEGIN("Apply Jacobi preconditioner");
        for (int i = 0; i != m_sweeps; ++i)
            Y.elementWiseMultiply(alpha, *m_diag_inv_damped, X, beta);
        L3STER_PROFILE_REGION_END("Apply Jacobi preconditioner");
    }

private:
    inline void init(const Teuchos::RCP< const tpetra_operator_t >& A);

    Teuchos::RCP< tpetra_vector_t > m_diag_inv_damped;
    val_t                           m_damping, m_threshold;
    int                             m_sweeps;
};

void NativeJacobiImpl::init(const Teuchos::RCP< const tpetra_operator_t >& A)
{
    // Note: this should be assert(A->hasDiagonal()), but Tpetra::CRSMatrix does not supply an overload :(
    try
    {
        A->getLocalDiagCopy(*m_diag_inv_damped);
    }
    catch (...)
    {
        util::throwingAssert(false, "Operator passed to NativeJacobiPreconditioner must be diagonal-aware");
    }
    const auto diag_view       = m_diag_inv_damped->getLocalViewHost(Tpetra::Access::ReadWrite);
    const auto diag_len        = diag_view.extent(0);
    const auto damping         = m_damping;   // Avoid implicit `this` capture
    const auto threshold       = m_threshold; // Avoid implicit `this` capture
    const auto diag_inv_kernel = KOKKOS_LAMBDA(const int i)
    {
        const auto  val       = diag_view(i, 0);
        const val_t sign      = val < 0 ? -1. : 1.;
        const auto  scale     = sign * damping;
        const auto  val_abs   = std::fabs(val);
        const auto  val_fixed = std::max(val_abs, threshold);
        diag_view(i, 0)       = scale / val_fixed;
    };
    Kokkos::parallel_for("Invert diagonal for Jacobi preconditioner", diag_len, diag_inv_kernel);
}

struct NativeJacobiPreconditioner
{
    struct Options
    {
        val_t damping   = 1.;
        val_t threshold = 0.;
        int   sweeps    = 1;

        using Preconditioner = NativeJacobiPreconditioner;
    };

    template < std::same_as< Teuchos::RCP< const tpetra_operator_t > > OpertorRCP > // Disable implicit upcast
    static auto create(const Options& opts, const OpertorRCP& op) -> Teuchos::RCP< tpetra_operator_t >
    {
        return util::makeTeuchosRCP< NativeJacobiImpl >(op, opts.damping, opts.threshold, opts.sweeps);
    }
};
static_assert(OperatorBasedPreconditioner_c< NativeJacobiPreconditioner >);

struct NativeRichardsonPreconditioner
{
    struct Options
    {
        val_t damping = 1.;
        int   sweeps  = 1;

        using Preconditioner = NativeRichardsonPreconditioner;
    };

    template < std::same_as< Teuchos::RCP< const tpetra_operator_t > > OpertorRCP > // Disable implicit upcast
    static auto create(const Options& opts, const OpertorRCP& op) -> Teuchos::RCP< tpetra_operator_t >
    {
        return util::makeTeuchosRCP< NativeRichardsonImpl >(op, opts.damping, opts.sweeps);
    }
};
static_assert(OperatorBasedPreconditioner_c< NativeRichardsonPreconditioner >);

} // namespace lstr::solvers

namespace lstr
{
using NativeRichardsonOpts = solvers::NativeRichardsonPreconditioner::Options;
using NativeJacobiOpts     = solvers::NativeJacobiPreconditioner::Options;
} // namespace lstr
#endif // L3STER_SOLVE_PRECONDITIONERS_HPP
