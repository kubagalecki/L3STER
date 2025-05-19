#ifndef L3STER_BELOSSOLVERS_HPP
#define L3STER_BELOSSOLVERS_HPP

#include "l3ster/solve/PreconditionerInterface.hpp"
#include "l3ster/solve/SolverInterface.hpp"
#include "l3ster/util/TrilinosUtils.hpp"

#ifdef L3STER_TRILINOS_HAS_BELOS

namespace lstr::solvers
{
class BelosSolverInterface
{
    using solver_t         = Belos::SolverManager< val_t, tpetra_multivector_t, tpetra_operator_t >;
    using linear_problem_t = Belos::LinearProblem< val_t, tpetra_multivector_t, tpetra_operator_t >;

public:
    template < PreconditionerOptions_c PrecOpts >
    BelosSolverInterface(std::string name, const IterSolverOpts& solver_opts, const PrecOpts& precond_opts)
        : m_solver_name{std::move(name)}, m_solver_opts{solver_opts}, m_precond_init{precond_opts}
    {}

    inline auto solve(const Teuchos::RCP< const tpetra_operator_t >&,
                      const Teuchos::RCP< const tpetra_multivector_t >&,
                      const Teuchos::RCP< tpetra_multivector_t >&) -> IterSolveResult;

private:
    inline static auto parseBelosOpts(const IterSolverOpts& opts) -> Teuchos::RCP< Teuchos::ParameterList >;

    inline void            init(const Teuchos::RCP< const tpetra_operator_t >&    A,
                                const Teuchos::RCP< const tpetra_multivector_t >& b,
                                const Teuchos::RCP< tpetra_multivector_t >&       x);
    inline IterSolveResult solveImpl();

    Teuchos::RCP< solver_t >          m_solver;
    Teuchos::RCP< linear_problem_t >  m_linear_problem;
    std::string                       m_solver_name;
    IterSolverOpts                    m_solver_opts;
    DeferredPreconditionerInitializer m_precond_init;
};

auto BelosSolverInterface::parseBelosOpts(const IterSolverOpts& opts) -> Teuchos::RCP< Teuchos::ParameterList >
{
    auto retval = util::makeTeuchosRCP< Teuchos::ParameterList >();
    retval->set("Maximum Iterations", opts.max_iters);
    retval->set("Maximum Restarts", opts.max_restarts);
    retval->set("Num Blocks", opts.restart_length);
    retval->set("Convergence Tolerance", opts.tol);
    retval->set("Output Frequency", opts.print_freq);
    retval->set("Block Size", opts.block_size);
    retval->set("Deflation Quorum", -1);

    const auto verbosity_belos_format =
        static_cast< Belos::MsgType >((opts.verbosity.warnings ? Belos::MsgType::Warnings : 0) |
                                      (opts.verbosity.summary ? Belos::MsgType::FinalSummary : 0) |
                                      (opts.verbosity.iter_details ? Belos::MsgType::IterationDetails : 0) |
                                      (opts.verbosity.timing ? Belos::MsgType::TimingDetails : 0));
    retval->set("Verbosity", verbosity_belos_format);
    return retval;
}

void BelosSolverInterface::init(const Teuchos::RCP< const tpetra_operator_t >&    A,
                                const Teuchos::RCP< const tpetra_multivector_t >& b,
                                const Teuchos::RCP< tpetra_multivector_t >&       x)
{
    const auto solver_params  = parseBelosOpts(m_solver_opts);
    auto       solver_factory = Belos::SolverFactory< val_t, tpetra_multivector_t, tpetra_operator_t >{};
    m_solver                  = solver_factory.create(m_solver_name, solver_params);
    m_linear_problem          = util::makeTeuchosRCP< linear_problem_t >(A, x, b);
    if (auto precond = m_precond_init(A); precond)
        m_linear_problem->setLeftPrec(std::move(precond));
    const auto setup_success = m_linear_problem->setProblem();
    util::throwingAssert(setup_success, "Failed to set up Belos::LinearProblem");
    m_solver->setProblem(m_linear_problem);
}

IterSolveResult BelosSolverInterface::solveImpl()
{
    m_solver->reset(Belos::Problem);
    m_solver->reset(Belos::RecycleSubspace);
    const auto solve_status = m_solver->solve();
    const auto converged    = solve_status == Belos::Converged;
    util::throwingAssert(not m_solver_opts.throw_on_fail or converged, "Solver failed to converge");
    const int  num_iters = m_solver->getNumIters();
    const auto tol       = m_solver->achievedTol();
    return {tol, num_iters};
}

auto BelosSolverInterface::solve(const Teuchos::RCP< const tpetra_operator_t >&    A,
                                 const Teuchos::RCP< const tpetra_multivector_t >& b,
                                 const Teuchos::RCP< tpetra_multivector_t >&       x) -> IterSolveResult
{
    util::throwingAssert(x->getNumVectors() == b->getNumVectors(),
                         "The LHS and RHS multivectors must have the same number of columns");
    if (not m_solver)
        init(A, b, x);
    return solveImpl();
}
} // namespace lstr::solvers

namespace lstr
{
struct CG : public solvers::BelosSolverInterface
{
    template < solvers::PreconditionerOptions_c PrecondOpts = typename solvers::NullPreconditioner::Options >
    CG(const IterSolverOpts& solver_opts = {}, const PrecondOpts& precond_opts = {})
        : BelosSolverInterface("Block CG", solver_opts, precond_opts)
    {}
};

struct Gmres : public solvers::BelosSolverInterface
{
    template < solvers::PreconditionerOptions_c PrecondOpts = typename solvers::NullPreconditioner::Options >
    Gmres(const IterSolverOpts& solver_opts = {}, const PrecondOpts& precond_opts = {})
        : BelosSolverInterface("Pseudoblock GMRES", solver_opts, precond_opts)
    {}
};
} // namespace lstr
#endif // L3STER_TRILINOS_HAS_BELOS
#endif // L3STER_BELOSSOLVERS_HPP
