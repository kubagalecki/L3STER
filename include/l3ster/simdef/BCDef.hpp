#ifndef L3STER_SIMSTRUCTURE_BCDEF_HPP
#define L3STER_SIMSTRUCTURE_BCDEF_HPP

#include "EquationDef.hpp"

namespace lstr::def
{
template < Space S, Time T, size_t N >
struct DirichletBC
{
    constexpr DirichletBC(Field< S, T >& field_, Support< N > support_, val_t value_)
        : field{&field_}, support{support_}, value{value_}
    {}
    constexpr DirichletBC(Field< S, T >&                                field_,
                          Support< N >                                  support_,
                          typename detail::SpaceTraits< S >::init_fun_t value_)
        : field{&field_}, support{support_}, value{value_}
    {}
    constexpr DirichletBC(Field< S, T >& field_, Support< N > support_, Field< S, Time::Stationary >& value_)
        : field{&field_}, support{support_}, value{&value_}
    {}

    Field< S, T >*                 field;
    Support< N >                   support;
    typename Field< S, T >::init_t value;
};

template < Space S, Time T, size_t... N >
struct DirichletBCSet
{
    constexpr DirichletBCSet(const DirichletBC< S, T, N >&... bcs_) : bcs{bcs_...} {}

    std::tuple< DirichletBC< S, T, N >... > bcs;
};

namespace detail
{
template < typename T >
constexpr bool is_dirichletbcset = false;

template < Space S, Time T, size_t... N >
constexpr bool is_dirichletbcset< DirichletBCSet< S, T, N... > > = true;
} // namespace detail

template < typename T >
concept DirichletBCSet_c = detail::is_dirichletbcset< T >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_BCDEF_HPP
