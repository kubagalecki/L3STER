#ifndef L3STER_SOLVE_SOLVERS_HPP
#define L3STER_SOLVE_SOLVERS_HPP

#include "l3ster/solve/SolverInterface.hpp"

#include "Amesos2.hpp"

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
    void solveImpl(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                   const Teuchos::RCP< const tpetra_multivector_t >& b,
                   const Teuchos::RCP< tpetra_multivector_t >&       x)
    {
        m_solver->numericFactorization().solve();
    }

private:
    Teuchos::RCP< Amesos2::Solver< tpetra_crsmatrix_t, tpetra_multivector_t > > m_solver;
};
} // namespace lstr::solvers
#endif // L3STER_SOLVE_SOLVERS_HPP
