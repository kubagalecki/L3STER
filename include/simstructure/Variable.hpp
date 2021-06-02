#ifndef L3STER_SIMSTRUCTURE_VARIABLE_HPP
#define L3STER_SIMSTRUCTURE_VARIABLE_HPP

#include "defs/Typedefs.h"
#include "utils/Common.hpp"

namespace lstr
{
enum class VariableSpaceType : std::uint_fast8_t
{
    Scalar1D,
    Scalar2D,
    Scalar3D,
    Vector2D,
    Vector3D
};

enum class VariableTimeType : std::uint_fast8_t
{
    Stationary,
    Transient,
    TransientReverse
};

template < VariableSpaceType SPACE_T, VariableTimeType TIME_T, size_t N_DOMS >
struct Variable
{
    template < std::convertible_to< d_id_t >... DT >
    requires(sizeof...(DT) == N_DOMS) constexpr Variable(EnumTag< SPACE_T >, EnumTag< TIME_T >, DT... dom_ids)
        : domains{dom_ids...}
    {}

    std::array< d_id_t, N_DOMS > domains;
};

template < VariableSpaceType SPACE_T, VariableTimeType TIME_T, std::convertible_to< d_id_t >... DT >
Variable(EnumTag< SPACE_T >, EnumTag< TIME_T >, DT...) -> Variable< SPACE_T, TIME_T, sizeof...(DT) >;

template < typename T >
inline constexpr bool is_variable_v = false;

template < VariableSpaceType SPACE_T, VariableTimeType TIME_T, size_t N_DOMS >
inline constexpr bool is_variable_v< Variable< SPACE_T, TIME_T, N_DOMS > > = true;

template < typename T >
concept variable_concept = is_variable_v< T >;

template < variable_concept... Vars >
struct VariableSet
{
    constexpr VariableSet(const Vars&... vars) : variables{vars...} {}

    std::tuple< Vars... > variables;
};

template < variable_concept... Vars >
VariableSet(const Vars&...) -> VariableSet< Vars... >;
} // namespace lstr
#endif // L3STER_SIMSTRUCTURE_VARIABLE_HPP
