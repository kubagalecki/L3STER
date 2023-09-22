#ifndef L3STER_BELOSSOLVERS_HPP
#define L3STER_BELOSSOLVERS_HPP

#include "l3ster/solve/SolverInterface.hpp"

// Disable diagnostics triggered by Trilinos
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wvolatile"
#endif

#include "BelosSolverFactory_Tpetra.hpp"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

namespace lstr::solvers
{
class BelosSolverInterface : public IterativeSolverInterface< BelosSolverInterface >
{
    using solver_t         = Belos::SolverManager< val_t, tpetra_multivector_t, tpetra_operator_t >;
    using linear_problem_t = Belos::LinearProblem< val_t, tpetra_multivector_t, tpetra_operator_t >;

public:
    template < PrecondType type >
    BelosSolverInterface(const IterSolverOpts&      solver_opts,
                         const PrecondOpts< type >& precond_opts,
                         std::string_view           name)
        : IterativeSolverInterface< BelosSolverInterface >(solver_opts, precond_opts), m_solver_name{name}
    {}

    inline void initializeImpl(const Teuchos::RCP< const tpetra_operator_t >&    A,
                               const Teuchos::RCP< const tpetra_multivector_t >& b,
                               const Teuchos::RCP< tpetra_multivector_t >&       x);
    inline void solveImpl(const Teuchos::RCP< const tpetra_operator_t >&,
                          const Teuchos::RCP< const tpetra_multivector_t >&,
                          const Teuchos::RCP< tpetra_multivector_t >&);

private:
    inline void initSolver();
    inline void initLinProblem(const Teuchos::RCP< const tpetra_operator_t >&    A,
                               const Teuchos::RCP< const tpetra_multivector_t >& b,
                               const Teuchos::RCP< tpetra_multivector_t >&       x);

    inline static auto parseBelosOpts(const IterSolverOpts& opts) -> Teuchos::RCP< Teuchos::ParameterList >;

    Teuchos::RCP< solver_t >         m_solver;
    Teuchos::RCP< linear_problem_t > m_linear_problem;
    std::string                      m_solver_name;
};

void BelosSolverInterface::initializeImpl(const Teuchos::RCP< const tpetra_operator_t >&    A,
                                          const Teuchos::RCP< const tpetra_multivector_t >& b,
                                          const Teuchos::RCP< tpetra_multivector_t >&       x)
{
    initSolver();
    initLinProblem(A, b, x);
    m_solver->setProblem(m_linear_problem);
}

void BelosSolverInterface::solveImpl(const Teuchos::RCP< const tpetra_operator_t >&,
                                     const Teuchos::RCP< const tpetra_multivector_t >&,
                                     const Teuchos::RCP< tpetra_multivector_t >&)
{
    const auto solve_status = m_solver->solve();
    util::throwingAssert(solve_status == Belos::Converged, "Solver failed to converge");
}

void BelosSolverInterface::initSolver()
{
    const auto solver_params  = parseBelosOpts(m_solver_opts);
    auto       solver_factory = Belos::SolverFactory< val_t, tpetra_multivector_t, tpetra_operator_t >{};
    m_solver                  = solver_factory.create(m_solver_name, solver_params);
}

void BelosSolverInterface::initLinProblem(const Teuchos::RCP< const tpetra_operator_t >&    A,
                                          const Teuchos::RCP< const tpetra_multivector_t >& b,
                                          const Teuchos::RCP< tpetra_multivector_t >&       x)
{
    m_linear_problem = util::makeTeuchosRCP< linear_problem_t >(A, x, b);
    if (const auto precond = m_preconditioner.get(); not precond.is_null())
        m_linear_problem->setLeftPrec(precond);
    const auto setup_success = m_linear_problem->setProblem();
    util::throwingAssert(setup_success, "Failed to set up Belos::LinearProblem");
}

auto BelosSolverInterface::parseBelosOpts(const IterSolverOpts& opts) -> Teuchos::RCP< Teuchos::ParameterList >
{
    auto retval = util::makeTeuchosRCP< Teuchos::ParameterList >();
    retval->set("Maximum Iterations", opts.max_iters);
    retval->set("Convergence Tolerance", opts.tol);
    retval->set("Output Frequency", opts.print_freq);
    retval->set("Block Size", opts.block_size);

    const auto verbosity_belos_format =
        static_cast< Belos::MsgType >((opts.verbosity.warnings ? Belos::MsgType::Warnings : 0) |
                                      (opts.verbosity.summary ? Belos::MsgType::FinalSummary : 0) |
                                      (opts.verbosity.iter_details ? Belos::MsgType::IterationDetails : 0) |
                                      (opts.verbosity.timing ? Belos::MsgType::TimingDetails : 0));
    retval->set("Verbosity", verbosity_belos_format);
    return retval;
}
} // namespace lstr::solvers

namespace lstr
{
struct CG : public solvers::BelosSolverInterface
{
    template < solvers::PrecondType type = solvers::PrecondType::None >
    CG(const IterSolverOpts& solver_opts = {}, const solvers::PrecondOpts< type >& precond_opts = {})
        : BelosSolverInterface(solver_opts, precond_opts, "Block CG")
    {}
};
} // namespace lstr
#endif // L3STER_BELOSSOLVERS_HPP
