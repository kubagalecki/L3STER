#ifndef L3STER_SOLVE_SOLVERS_HPP
#define L3STER_SOLVE_SOLVERS_HPP

#include "l3ster/solve/SolverInterface.hpp"

// Disable diagnostics triggered by Trilinos
#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wreturn-type"
#pragma GCC diagnostic ignored "-Wconversion"
#pragma GCC diagnostic ignored "-Wfloat-conversion"
#pragma GCC diagnostic ignored "-Wvolatile"
#endif

#include "Amesos2.hpp"
#include "BelosSolverFactory_Tpetra.hpp"
#include "Ifpack2_Factory.hpp"

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

namespace lstr::solvers
{
class Lapack : public SolverInterface< Lapack >
{
public:
    void initializeImpl(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                        const Teuchos::RCP< const tpetra_multivector_t >& b,
                        const Teuchos::RCP< tpetra_multivector_t >&       x)
    {
        m_solver = Amesos2::create("Lapack", A, x, b);
        util::throwingAssert(m_solver->matrixShapeOK(), "Incompatible matrix/vector dimensions");
        m_solver->preOrdering().symbolicFactorization();
    }
    void solveImpl(const Teuchos::RCP< const tpetra_crsmatrix_t >&,
                   const Teuchos::RCP< const tpetra_multivector_t >&,
                   const Teuchos::RCP< tpetra_multivector_t >&)
    {
        m_solver->numericFactorization().solve();
    }

private:
    Teuchos::RCP< Amesos2::Solver< tpetra_crsmatrix_t, tpetra_multivector_t > > m_solver;
};

class CG : public SolverInterface< CG >
{
    using precond_t        = decltype(Ifpack2::Factory{}.create("", Teuchos::RCP< const tpetra_crsmatrix_t >{}));
    using solver_t         = decltype(Belos::SolverFactory< val_t, tpetra_multivector_t, tpetra_operator_t >{}.create(
        "", Teuchos::RCP< Teuchos::ParameterList >{}));
    using linear_problem_t = Teuchos::RCP< Belos::LinearProblem< val_t, tpetra_multivector_t, tpetra_operator_t > >;

public:
    CG(double tol = 1e-6, int max_iters = 10'000, Belos::MsgType verbosity = Belos::Warnings, int print_freq = 100)
        : m_solver_params{util::makeTeuchosRCP< Teuchos::ParameterList >()}
    {
        m_solver_params->set("Block Size", 1);
        m_solver_params->set("Maximum Iterations", max_iters);
        m_solver_params->set("Convergence Tolerance", tol);
        m_solver_params->set("Verbosity", verbosity);
        m_solver_params->set("Output Frequency", print_freq);
    }

    void initializeImpl(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                        const Teuchos::RCP< const tpetra_multivector_t >& b,
                        const Teuchos::RCP< tpetra_multivector_t >&       x)
    {
        auto precond_factory = Ifpack2::Factory{};
        m_precond            = precond_factory.create("CHEBYSHEV", A);
        auto precond_params  = util::makeTeuchosRCP< Teuchos::ParameterList >();
        precond_params->set("chebyshev: degree", 3);
        m_precond->setParameters(*precond_params);
        m_precond->initialize();
        util::throwingAssert(m_precond->isInitialized(), "Failed to initialize the preconditioner");
        m_precond->compute();

        m_linear_problem =
            util::makeTeuchosRCP< Belos::LinearProblem< val_t, tpetra_multivector_t, tpetra_operator_t > >(A, x, b);
        m_linear_problem->setLeftPrec(m_precond);
        util::throwingAssert(m_linear_problem->setProblem(), "Failed to set up Belos::LinearProblem");

        auto solver_factory = Belos::SolverFactory< val_t, tpetra_multivector_t, tpetra_operator_t >{};
        m_solver            = solver_factory.create("Block CG", m_solver_params);
        m_solver->setProblem(m_linear_problem);
    }
    void solveImpl(const Teuchos::RCP< const tpetra_crsmatrix_t >&,
                   const Teuchos::RCP< const tpetra_multivector_t >&,
                   const Teuchos::RCP< tpetra_multivector_t >&)
    {
        m_precond->compute();
        util::throwingAssert(m_solver->solve() == Belos::Converged, "Solver failed to converge");
    }

private:
    precond_t                              m_precond;
    solver_t                               m_solver;
    linear_problem_t                       m_linear_problem;
    Teuchos::RCP< Teuchos::ParameterList > m_solver_params;
};
} // namespace lstr::solvers
#endif // L3STER_SOLVE_SOLVERS_HPP
