#ifndef L3STER_SOLVE_SOLVERINTERFACE_HPP
#define L3STER_SOLVE_SOLVERINTERFACE_HPP

#include "l3ster/solve/PreconditionerManager.hpp"

#include <concepts>
#include <type_traits>

namespace lstr
{
struct SolverVerbosity
{
    bool warnings     = true;
    bool summary      = true;
    bool iter_details = false;
    bool timing       = false;
};

struct IterSolverOpts
{
    double          tol           = 1e-6;   // Iterative solver tolerance
    int             max_iters     = 10'000; // Max number of iterations
    bool            throw_on_fail = true;   // Whether to throw if convergence exceeds the tolerance
    SolverVerbosity verbosity     = {};     // Verbosity level (see above)
    int             print_freq    = 10;     // How often to print the convergence information
    int             block_size    = 1;      // Block size is currently always 1, this is future-proofing
};

struct IterSolveInfo
{
    val_t tol;
    int   num_iters;
};

namespace solvers
{
template < typename CRTP >
class DirectSolverInterface
{
public:
    inline void solve(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                      const Teuchos::RCP< const tpetra_multivector_t >& b,
                      const Teuchos::RCP< tpetra_multivector_t >&       x);

private:
    bool is_initialized = false;
};

template < typename T >
concept DirectSolver_c =
    requires(T solver) { std::invoke([]< typename B >(const DirectSolverInterface< B >&) {}, solver); };

template < typename CRTP >
class IterativeSolverInterface
{
public:
    template < PrecondType precond_type >
    IterativeSolverInterface(const IterSolverOpts& solver_opts, const PrecondOpts< precond_type >& precond_opts)
        : m_solver_opts{solver_opts}, m_preconditioner{precond_opts}
    {}

    inline IterSolveInfo solve(const Teuchos::RCP< const tpetra_operator_t >&    A,
                               const Teuchos::RCP< const tpetra_multivector_t >& b,
                               const Teuchos::RCP< tpetra_multivector_t >&       x);

protected:
    IterSolverOpts        m_solver_opts;
    PreconditionerManager m_preconditioner;

private:
    bool m_is_initialized = false;
};

template < typename T >
concept IterativeSolver_c =
    requires(T solver) { std::invoke([]< typename B >(const IterativeSolverInterface< B >&) {}, solver); };

template < typename T >
concept Solver_c = DirectSolver_c< T > or IterativeSolver_c< T >;

template < typename CRTP >
void DirectSolverInterface< CRTP >::solve(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                                          const Teuchos::RCP< const tpetra_multivector_t >& b,
                                          const Teuchos::RCP< tpetra_multivector_t >&       x)
{
    util::throwingAssert(x->getNumVectors() == b->getNumVectors(),
                         "The LHS and RHS multivectors must have the same number of columns");

    if (not is_initialized)
        static_cast< CRTP* >(this)->initializeImpl(A, b, x);
    is_initialized = true;
    static_cast< CRTP* >(this)->solveImpl(A, b, x);
}

template < typename CRTP >
IterSolveInfo IterativeSolverInterface< CRTP >::solve(const Teuchos::RCP< const tpetra_operator_t >&    A,
                                                      const Teuchos::RCP< const tpetra_multivector_t >& b,
                                                      const Teuchos::RCP< tpetra_multivector_t >&       x)
{
    util::throwingAssert(x->getNumVectors() == b->getNumVectors(),
                         "The LHS and RHS multivectors must have the same number of columns");

    if (not m_is_initialized)
    {
        m_preconditioner.initialize(A);
        m_preconditioner.compute();
        static_cast< CRTP* >(this)->initializeImpl(A, b, x);
        m_is_initialized = true;
    }
    m_preconditioner.compute();
    return static_cast< CRTP* >(this)->solveImpl(A, b, x);
}
} // namespace solvers
} // namespace lstr
#endif // L3STER_SOLVE_SOLVERINTERFACE_HPP
