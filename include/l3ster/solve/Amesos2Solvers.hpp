#ifndef L3STER_AMESOS2SOLVERS_HPP
#define L3STER_AMESOS2SOLVERS_HPP

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

#if defined(__GNUC__) || defined(__GNUG__)
#pragma GCC diagnostic pop
#endif

namespace lstr::solvers
{
class Amesos2SolverInterface : public DirectSolverInterface< Amesos2SolverInterface >
{
public:
    Amesos2SolverInterface(std::string_view name) : m_name{name} {}

    void initializeImpl(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                        const Teuchos::RCP< const tpetra_multivector_t >& b,
                        const Teuchos::RCP< tpetra_multivector_t >&       x)
    {
        m_solver = Amesos2::create(m_name, A, x, b);
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
    std::string                                                                 m_name;
};

class KLU2 : public Amesos2SolverInterface
{
public:
    KLU2() : Amesos2SolverInterface("KLU2") {}
};

class Lapack : public Amesos2SolverInterface
{
public:
    Lapack() : Amesos2SolverInterface("Lapack") {}
};
} // namespace lstr::solvers
#endif // L3STER_AMESOS2SOLVERS_HPP
