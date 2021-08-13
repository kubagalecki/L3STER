#ifndef L3STER_SIMSTRUCTURE_SUBITERDEF_HPP
#define L3STER_SIMSTRUCTURE_SUBITERDEF_HPP

#include "simstructure/ConvergenceDef.hpp"
#include "simstructure/SystemDef.hpp"

namespace lstr::def
{
template < System_c S, Convergence_c C = Convergence< std::nullptr_t, ConvergenceTypes::FixedIter > >
struct Subiter
{
    constexpr Subiter(const S& system_, const C& conv_ = {}) : system{system_}, conv{conv_} {}

    S system;
    C conv;
};

namespace detail
{
template < typename T >
constexpr bool is_subiter = false;

template < System_c S, Convergence_c C >
constexpr bool is_subiter< Subiter< S, C > > = true;
} // namespace detail

template < typename T >
concept Subiter_c = detail::is_subiter< T >;

template < Subiter_c... S >
struct SubiterSet
{
    constexpr SubiterSet(const S&... subiters_) : subiters{subiters_...} {}

    std::tuple< S... > subiters;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_SUBITERDEF_HPP
