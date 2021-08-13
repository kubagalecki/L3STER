#ifndef L3STER_SIMSTRUCTURE_SYSTEMDEF_HPP
#define L3STER_SIMSTRUCTURE_SYSTEMDEF_HPP

#include "simstructure/AlgebraicSolverDef.hpp"
#include "simstructure/PhysicsDef.hpp"

namespace lstr::def
{
template < AlgebraicSolverTypes AST, Physics_c... P >
struct System
{
    constexpr System(const AlgebraicSolver< AST >& solver_, const P&... physics_)
        : solver{solver_}, physics{physics_...}
    {}

    AlgebraicSolver< AST > solver;
    std::tuple< P... >     physics;
};

namespace detail
{
template < typename T >
constexpr bool is_system = false;

template < AlgebraicSolverTypes AST, Physics_c... P >
constexpr bool is_system< System< AST, P... > > = true;
} // namespace detail

template < typename T >
concept System_c = detail::is_system< T >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_SYSTEMDEF_HPP
