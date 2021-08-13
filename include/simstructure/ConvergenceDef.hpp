#ifndef L3STER_SIMSTRUCTURE_CONVERGENCEDEF_HPP
#define L3STER_SIMSTRUCTURE_CONVERGENCEDEF_HPP

#include "simstructure/PhysicsDef.hpp"

#include <limits>

namespace lstr::def
{
enum struct ConvergenceTypes
{
    L2,
    LInf,
    FixedIter
};

template < ConvergenceTypes C >
struct ConvergenceParams;

template <>
struct ConvergenceParams< ConvergenceTypes::FixedIter >
{
    size_t max_iters = 1;
};

template < ConvergenceTypes C >
requires(C == ConvergenceTypes::L2 or C == ConvergenceTypes::LInf) struct ConvergenceParams< C >
{
    val_t  tol       = 1e-8;
    size_t max_iters = std::numeric_limits< size_t >::max();
};

template < typename P, ConvergenceTypes CONV >
struct Convergence;

template <>
struct Convergence< std::nullptr_t, ConvergenceTypes::FixedIter >
{
    constexpr Convergence(const ConvergenceParams< ConvergenceTypes::FixedIter >& params_ = {}) : params{params_} {}

    ConvergenceParams< ConvergenceTypes::FixedIter > params;
};

Convergence() -> Convergence< std::nullptr_t, ConvergenceTypes::FixedIter >;
Convergence(ConvergenceParams< ConvergenceTypes::FixedIter >)
    -> Convergence< std::nullptr_t, ConvergenceTypes::FixedIter >;

template < Physics_c P, ConvergenceTypes C >
struct Convergence< P, C >
{
    constexpr Convergence(const P& physics_, const ConvergenceParams< C >& params_ = {})
        : physics{physics_}, params{params_}
    {}

    P                      physics;
    ConvergenceParams< C > params;
};

namespace detail
{
template < typename T >
constexpr bool is_convergence = false;

template < typename P, ConvergenceTypes C >
constexpr bool is_convergence< Convergence< P, C > > = true;
} // namespace detail

template < typename T >
concept Convergence_c = detail::is_convergence< T >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_CONVERGENCEDEF_HPP
