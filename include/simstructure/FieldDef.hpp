#ifndef L3STER_SIMSTRUCTURE_FIELDDEF_HPP
#define L3STER_SIMSTRUCTURE_FIELDDEF_HPP

#include "simstructure/TimeDef.hpp"

#include <array>
#include <concepts>
#include <variant>

namespace lstr::def
{
enum struct Space
{
    D2,
    D3
};

enum struct Time
{
    Stationary,
    Transient,
    TransientReverse
};

template < Space S, Time T >
struct Field
{
    static constexpr Space space_v = S;
    static constexpr Time  time_v  = T;
};

namespace detail
{
template < Space >
struct SpaceTraits;

template <>
struct SpaceTraits< Space::D2 >
{
    using init_fun_t = val_t (*)(val_t, val_t);
};

template <>
struct SpaceTraits< Space::D3 >
{
    using init_fun_t = val_t (*)(val_t, val_t, val_t);
};
} // namespace detail

template < Space S, Time T >
requires(T == Time::Transient || T == Time::TransientReverse) struct Field< S, T >
{
    using init_t = std::variant< val_t, typename detail::SpaceTraits< S >::init_fun_t, Field< S, Time::Stationary >* >;
    static constexpr Space space_v = S;
    static constexpr Time  time_v  = T;

    constexpr Field(const Timeline* time_, val_t init_) : time{time_}, init{init_} {}
    constexpr Field(const Timeline* time_, typename detail::SpaceTraits< S >::init_fun_t init_)
        : time{time_}, init{init_}
    {}
    constexpr Field(const Timeline* time_, const Field< S, Time::Stationary >* init_) : time{time_}, init{init_} {}

    const Timeline* time;
    init_t          init;
};

template < Space S, Time T, size_t N >
struct FieldSet
{
    template < std::same_as< const Field< S, T >* >... F >
    requires(sizeof...(F) == N) constexpr FieldSet(F... fields_) : fields{fields_...} {}

    std::array< const Field< S, T >*, N > fields;
};

template < typename... F >
FieldSet(F&...) -> FieldSet< std::tuple_element_t< 0, std::tuple< F... > >::space_v,
                             std::tuple_element_t< 0, std::tuple< F... > >::time_v,
                             sizeof...(F) >;

template < size_t N >
struct Support
{
    template < std::convertible_to< d_id_t >... D >
    requires(sizeof...(D) == N) constexpr Support(D... doms) : domains{doms...} {}

    std::array< d_id_t, N > domains;
};

template < std::convertible_to< d_id_t >... D >
Support(D...) -> Support< sizeof...(D) >;
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_FIELDDEF_HPP
