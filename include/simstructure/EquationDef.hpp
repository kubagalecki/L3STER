#ifndef L3STER_SIMSTRUCTURE_EQUATIONDEF_HPP
#define L3STER_SIMSTRUCTURE_EQUATIONDEF_HPP

#include "simstructure/FieldDef.hpp"
#include "simstructure/TimeSchemeDef.hpp"

#include <tuple>

namespace lstr::def
{
template < typename Kernel, Space S, Time T, TimeSchemes TS, size_t NFIELDS, size_t SUPSIZE >
requires(not(T == Time::Stationary xor TS == TimeSchemes::None)) struct Equation
{
    using support_t  = Support< SUPSIZE >;
    using fieldset_t = FieldSet< S, T, NFIELDS >;

    constexpr Equation(Kernel& kernel_, support_t support_, fieldset_t fields_) requires(T == Time::Stationary)
        : kernel{&kernel_}, support{support_}, fields{fields_}, time_solver{}
    {}
    constexpr Equation(Kernel           kernel_,
                       support_t        support_,
                       fieldset_t       fields_,
                       TimeSolver< TS > time_solver_) requires(T != Time::Stationary)
        : kernel{&kernel_}, support{support_}, fields{fields_}, time_solver{time_solver_}
    {}

    Kernel*          kernel;
    support_t        support;
    fieldset_t       fields;
    TimeSolver< TS > time_solver;
};

template < typename Kernel, Space S, Time T, size_t NFIELDS, size_t SUPSIZE >
Equation(Kernel&, Support< SUPSIZE >, FieldSet< S, T, NFIELDS >)
    -> Equation< Kernel, S, T, TimeSchemes::None, NFIELDS, SUPSIZE >;

namespace detail
{
template < typename T >
constexpr bool is_equation = false;

template < typename Kernel, Space S, Time T, TimeSchemes TS, size_t NFIELDS, size_t SUPSIZE >
constexpr bool is_equation< Equation< Kernel, S, T, TS, NFIELDS, SUPSIZE > > = true;
} // namespace detail

template < typename T >
concept Equation_c = detail::is_equation< T >;

template < Equation_c... E >
struct EquationSet
{
    constexpr EquationSet(const E&... equations_) : equations{equations_...} {}

    std::tuple< E... > equations;
};

namespace detail
{
template < typename T >
constexpr bool is_equationset = false;

template < Equation_c... E >
constexpr bool is_equationset< EquationSet< E... > > = true;
} // namespace detail

template < typename T >
concept EquationSet_c = detail::is_equationset< T >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_EQUATIONDEF_HPP
