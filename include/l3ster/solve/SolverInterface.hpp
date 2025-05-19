#ifndef L3STER_SOLVE_SOLVERINTERFACE_HPP
#define L3STER_SOLVE_SOLVERINTERFACE_HPP

#include "l3ster/common/TrilinosTypedefs.h"

#include <concepts>

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
    double          tol            = 1e-6;   // Iterative solver tolerance
    int             max_iters      = 10'000; // Max number of iterations
    int             max_restarts   = 39;     // Max number of restarts (relevant for GMRES)
    int             restart_length = 250;    // Number of iterations before restart (relevant for GMRES)
    bool            throw_on_fail  = true;   // Whether to throw if convergence exceeds the tolerance
    SolverVerbosity verbosity      = {};     // Verbosity level (see above)
    int             print_freq     = 10;     // How often to print the convergence information
    int             block_size     = 1;      // Block size is currently always 1, this is future-proofing
};

struct IterSolveResult
{
    val_t tol;
    int   num_iters;
};

struct DirectSolveResult
{};

namespace solvers
{
template < typename Solver >
concept DirectSolver_c = requires(Solver                                            solver,
                                  const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                                  const Teuchos::RCP< const tpetra_multivector_t >& b,
                                  const Teuchos::RCP< tpetra_multivector_t >&       x) {
    { solver.solve(A, b, x) } -> std::same_as< DirectSolveResult >;
};

template < typename Solver >
concept IterativeSolver_c = requires(Solver                                            solver,
                                     const Teuchos::RCP< const tpetra_operator_t >&    A,
                                     const Teuchos::RCP< const tpetra_multivector_t >& b,
                                     const Teuchos::RCP< tpetra_multivector_t >&       x) {
    { solver.solve(A, b, x) } -> std::same_as< IterSolveResult >;
};

template < typename T >
concept Solver_c = DirectSolver_c< T > or IterativeSolver_c< T >;
} // namespace solvers
} // namespace lstr
#endif // L3STER_SOLVE_SOLVERINTERFACE_HPP
