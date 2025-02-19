#ifndef L3STER_AMESOS2SOLVERS_HPP
#define L3STER_AMESOS2SOLVERS_HPP

#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/solve/SolverInterface.hpp"

#ifdef L3STER_TRILINOS_HAS_AMESOS2

namespace lstr
{
namespace solvers
{
class Amesos2SolverInterface
{
public:
    Amesos2SolverInterface(std::string name) : m_name{std::move(name)} {}

    auto solve(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
               const Teuchos::RCP< const tpetra_multivector_t >& b,
               const Teuchos::RCP< tpetra_multivector_t >&       x) -> DirectSolveResult
    {
        util::throwingAssert(x->getNumVectors() == b->getNumVectors(),
                             "The LHS and RHS multivectors must have the same number of columns");
        if (not m_solver)
        {
            m_solver = Amesos2::create(m_name, A, x, b);
            util::throwingAssert(m_solver->matrixShapeOK(), "Incompatible matrix/vector dimensions");
            m_solver->preOrdering().symbolicFactorization();
        }
        m_solver->numericFactorization().solve();
        return {};
    }

private:
    Teuchos::RCP< Amesos2::Solver< tpetra_crsmatrix_t, tpetra_multivector_t > > m_solver;
    std::string                                                                 m_name;
};
} // namespace solvers

class Klu2 : public solvers::Amesos2SolverInterface
{
public:
    Klu2() : Amesos2SolverInterface("KLU2") {}
};

class Lapack : public solvers::Amesos2SolverInterface
{
public:
    Lapack() : Amesos2SolverInterface("Lapack") {}
};
} // namespace lstr
#endif // L3STER_TRILINOS_HAS_AMESOS2
#endif // L3STER_AMESOS2SOLVERS_HPP
