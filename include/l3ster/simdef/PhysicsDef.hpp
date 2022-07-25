#ifndef L3STER_SIMSTRUCTURE_PHYSICSDEF_HPP
#define L3STER_SIMSTRUCTURE_PHYSICSDEF_HPP

#include "BCDef.hpp"
#include "EquationDef.hpp"

namespace lstr::def
{
template < Equation_c Kernel, EquationSet_c EBCs, DirichletBCSet_c DBCs >
struct Physics
{
    constexpr Physics(const Kernel& kernel_, const EBCs& ebcs_, const DBCs& dbcs_)
        : kernel{kernel_}, ebcs{ebcs_}, dbcs{dbcs_}
    {}

    Kernel kernel;
    EBCs   ebcs;
    DBCs   dbcs;
};

namespace detail
{
template < typename T >
constexpr bool is_physics = false;

template < Equation_c Kernel, EquationSet_c EBCs, DirichletBCSet_c DBCs >
constexpr bool is_physics< Physics< Kernel, EBCs, DBCs > > = true;
} // namespace detail

template < typename T >
concept Physics_c = detail::is_physics< T >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_PHYSICSDEF_HPP
