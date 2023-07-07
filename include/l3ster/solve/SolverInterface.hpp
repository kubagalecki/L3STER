#ifndef L3STER_SOLVE_SOLVERINTERFACE_HPP
#define L3STER_SOLVE_SOLVERINTERFACE_HPP

#include "l3ster/util/TrilinosUtils.hpp"

#include <concepts>
#include <type_traits>

namespace lstr::solvers
{
template < typename Derived >
    requires std::is_class_v< Derived > and std::same_as< Derived, std::remove_cv_t< Derived > >
class SolverInterface
{
public:
    void solve(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
               const Teuchos::RCP< const tpetra_multivector_t >& b,
               const Teuchos::RCP< tpetra_multivector_t >&       x)
    {
        if (not is_initialized)
            static_cast< Derived* >(this)->initializeImpl(A, b, x);
        is_initialized = true;
        static_cast< Derived* >(this)->solveImpl(A, b, x);
    }

private:
    bool is_initialized = false;
};

template < typename T >
concept Solver_c = requires(T solver) {
    []< typename B >(const SolverInterface< B >&) {
    }(solver);
};
} // namespace lstr::solvers
#endif // L3STER_SOLVE_SOLVERINTERFACE_HPP
