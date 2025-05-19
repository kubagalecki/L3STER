#ifndef L3STER_SOLVE_IFPACK2PRECONDITIONERS_HPP
#define L3STER_SOLVE_IFPACK2PRECONDITIONERS_HPP

#include "l3ster/solve/PreconditionerInterface.hpp"

#ifdef L3STER_TRILINOS_HAS_IFPACK2

namespace lstr::solvers
{
namespace detail
{
template < typename Opts >
auto makeIfpack2RelaxOpts(const char* name, const Opts& opts) -> Teuchos::ParameterList
{
    auto retval = Teuchos::ParameterList{};
    retval.set("relaxation: type", name);
    retval.set("relaxation: sweeps", opts.sweeps);
    retval.set("relaxation: damping factor", opts.damping);
    retval.set("relaxation: use l1", opts.enable_l1);
    retval.set("relaxation: l1 eta", opts.l1_eta);
    retval.set("relaxation: fix tiny diagonal entries", opts.diag_threshold != 0.);
    retval.set("relaxation: min diagonal value", opts.diag_threshold);
    return retval;
}

template < typename Opts >
auto makeIfpack2ChebyshevOpts(const Opts& opts) -> Teuchos::ParameterList
{
    auto retval = Teuchos::ParameterList{};
    retval.set("chebyshev: degree", opts.degree);
    retval.set("chebyshev: ratio eigenvalue", opts.cond_est);
    retval.set("chebyshev: eigenvalue max iterations", opts.max_power_iters);
    retval.set("chebyshev: min diagonal value", opts.diag_threshold);
    retval.set("chebyshev: boost factor", opts.boost_factor);
    return retval;
}

template < typename Opts >
auto makeIfpack2RilukOpts(const Opts& opts) -> Teuchos::ParameterList
{
    auto retval = Teuchos::ParameterList{};
    retval.set("fact: iluk level-of-fill", opts.level_of_fill);
    retval.set("fact: relax value", opts.relax_value);
    retval.set("fact: absolute threshold", opts.absolute_threshold);
    retval.set("fact: relative threshold", opts.relative_threshold);
    return retval;
}

template < typename Options >
auto createIfpack2RelaxImpl(const char*                                     name,
                            const Options&                                  opts,
                            const Teuchos::RCP< const tpetra_crsmatrix_t >& matrix) -> Teuchos::RCP< tpetra_operator_t >
{
    auto       retval = Ifpack2::Factory::create("RELAXATION", matrix);
    const auto params = makeIfpack2RelaxOpts(name, opts);
    retval->setParameters(params);
    retval->initialize();
    util::throwingAssert(retval->isInitialized(), "Failed to initialize Ifpack2 preconditioner");
    retval->compute();
    return retval;
}
} // namespace detail

template < typename CRTP >
struct Ifpack2RelaxPrecondBase
{
    using Base = Ifpack2RelaxPrecondBase;
    struct Options
    {
        int   sweeps         = 1;
        val_t damping        = 1.;
        bool  enable_l1      = false;
        val_t l1_eta         = 1.5;
        val_t diag_threshold = 0.;

        using Preconditioner = CRTP;
    };
    template < std::same_as< Teuchos::RCP< const tpetra_crsmatrix_t > > MatrixRCP > // Disable implicit upcast
    static auto create(const Options& opts, const MatrixRCP& matrix) -> Teuchos::RCP< tpetra_operator_t >
    {
        return detail::createIfpack2RelaxImpl(CRTP::ifpack2_name, opts, matrix);
    }
};

struct Ifpack2RichardsonPreconditioner : Ifpack2RelaxPrecondBase< Ifpack2RichardsonPreconditioner >
{
    static constexpr auto ifpack2_name = "Richardson";
    using Options                      = Base::Options;
};

struct Ifpack2JacobiPreconditioner : Ifpack2RelaxPrecondBase< Ifpack2JacobiPreconditioner >
{
    static constexpr auto ifpack2_name = "Jacobi";
    using Options                      = Base::Options;
};

struct Ifpack2SGSPreconditioner : Ifpack2RelaxPrecondBase< Ifpack2SGSPreconditioner >
{
    static constexpr auto ifpack2_name = "Symmetric Gauss-Seidel";
    using Options                      = Base::Options;
};

static_assert(MatrixBasedPreconditioner_c< Ifpack2RichardsonPreconditioner >);
static_assert(MatrixBasedPreconditioner_c< Ifpack2JacobiPreconditioner >);
static_assert(MatrixBasedPreconditioner_c< Ifpack2SGSPreconditioner >);

struct Ifpack2ChebyshevPreconditioner
{
    struct Options
    {
        int   degree          = 1;   // Chebyshev polynomial degree
        val_t cond_est        = 30.; // Estimated ratio of max/min eigenvalues
        int   max_power_iters = 10;  // Max number of iterations of power method for estimating max eigenvalue
        val_t boost_factor    = 1.1; // Factor for max eigenvalue estimation to ensure margin
        val_t diag_threshold  = 0.;  // Diagonal entries are bounded from below by this value

        using Preconditioner = Ifpack2ChebyshevPreconditioner;
    };

    template < std::same_as< Teuchos::RCP< const tpetra_crsmatrix_t > > MatrixRCP > // Disable implicit upcast
    static auto create(const Options& opts, const MatrixRCP& matrix) -> Teuchos::RCP< tpetra_operator_t >
    {
        auto       retval = Ifpack2::Factory::create("CHEBYSHEV", matrix);
        const auto params = detail::makeIfpack2ChebyshevOpts(opts);
        retval->setParameters(params);
        retval->initialize();
        util::throwingAssert(retval->isInitialized(), "Failed to initialize Ifpack2 preconditioner");
        retval->compute();
        return retval;
    }
};
static_assert(MatrixBasedPreconditioner_c< Ifpack2ChebyshevPreconditioner >);

struct Ifpack2Riluk
{
    struct Options
    {
        int   level_of_fill      = 0;
        val_t absolute_threshold = 0.;
        val_t relative_threshold = 1.;
        val_t relax_value        = 0.;

        using Preconditioner = Ifpack2Riluk;
    };
    template < std::same_as< Teuchos::RCP< const tpetra_crsmatrix_t > > MatrixRCP > // Disable implicit upcast
    static auto create(const Options& opts, const MatrixRCP& matrix) -> Teuchos::RCP< tpetra_operator_t >
    {
        auto       retval = Ifpack2::Factory::create("RILUK", matrix);
        const auto params = detail::makeIfpack2RilukOpts(opts);
        retval->setParameters(params);
        retval->initialize();
        util::throwingAssert(retval->isInitialized(), "Failed to initialize Ifpack2 preconditioner");
        retval->compute();
        return retval;
    }
};
static_assert(MatrixBasedPreconditioner_c< Ifpack2Riluk >);

struct Ifpack2Ilut
{
    struct Options
    {
        int   level_of_fill      = 1;
        val_t drop_tolerance     = 0.;
        val_t absolute_threshold = 0.;
        val_t relative_threshold = 1.;
        val_t relax_value        = 0.;

        using Preconditioner = Ifpack2Ilut;
    };
    template < std::same_as< Teuchos::RCP< const tpetra_crsmatrix_t > > MatrixRCP > // Disable implicit upcast
    static auto create(const Options& opts, const MatrixRCP& matrix) -> Teuchos::RCP< tpetra_operator_t >
    {
        auto retval = Ifpack2::Factory::create("ILUT", matrix);
        auto params = detail::makeIfpack2RilukOpts(opts);
        params.set("fact: drop tolerance", opts.drop_tolerance);
        retval->setParameters(params);
        retval->initialize();
        util::throwingAssert(retval->isInitialized(), "Failed to initialize Ifpack2 preconditioner");
        retval->compute();
        return retval;
    }
};
static_assert(MatrixBasedPreconditioner_c< Ifpack2Ilut >);
} // namespace lstr::solvers

namespace lstr
{
using Ifpack2RichardsonOpts = solvers::Ifpack2RichardsonPreconditioner::Options;
using Ifpack2JacobiOpts     = solvers::Ifpack2JacobiPreconditioner::Options;
using Ifpack2SGSOpts        = solvers::Ifpack2SGSPreconditioner::Options;
using Ifpack2ChebyshevOpts  = solvers::Ifpack2ChebyshevPreconditioner::Options;
using Ifpack2IluKOpts       = solvers::Ifpack2Riluk::Options;
using Ifpack2IluTOpts       = solvers::Ifpack2Ilut::Options;
} // namespace lstr
#endif // L3STER_TRILINOS_HAS_IFPACK2
#endif // L3STER_SOLVE_IFPACK2PRECONDITIONERS_HPP
