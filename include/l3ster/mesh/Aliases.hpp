#ifndef L3STER_MESH_ALIASES_HPP
#define L3STER_MESH_ALIASES_HPP

#include "ElementTypes.hpp"
#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/Meta.hpp"

#include <functional>
#include <variant>

namespace lstr
{
template < template < typename... > typename T, template < ElementTypes, el_o_t > typename U >
using parametrize_type_over_element_types_and_orders_t = cart_prod_t< U, T, element_types, element_orders >;
template < ElementTypes ET, el_o_t EO >
class Element;
template < ElementTypes ET, el_o_t EO >
using element_ptr_t = Element< ET, EO >*;
template < ElementTypes ET, el_o_t EO >
using element_cptr_t         = const Element< ET, EO >*;
using element_ptr_variant_t  = parametrize_type_over_element_types_and_orders_t< std::variant, element_ptr_t >;
using element_cptr_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, element_cptr_t >;

template < ElementTypes TYPE, el_o_t ORDER >
using type_order_set          = ValuePack< TYPE, ORDER >;
using type_order_combinations = parametrize_type_over_element_types_and_orders_t< TypePack, type_order_set >;

namespace detail
{
template < typename F, bool is_const, typename... Args >
struct is_invocable_on_elements
{
    template < ElementTypes ET, el_o_t EO >
    struct invocable_on_element :
        std::conditional_t<
            std::is_invocable_v< F,
                                 std::conditional_t< is_const, const Element< ET, EO >&, Element< ET, EO >& >,
                                 Args... >,
            std::true_type,
            std::false_type >
    {};

    using invocability_set = parametrize_type_over_element_types_and_orders_t< TypePack, invocable_on_element >;

    template < typename >
    struct check_all;
    template < typename... T >
    struct check_all< TypePack< T... > >
    {
        static constexpr bool value = (T::value && ...);
    };

    static constexpr bool value = check_all< invocability_set >::value;
};
} // namespace detail

template < typename T >
concept invocable_on_elements = detail::is_invocable_on_elements< T, false >::value;
template < typename T >
concept invocable_on_const_elements = detail::is_invocable_on_elements< T, true >::value;
template < typename T, typename... Args >
concept invocable_on_elements_and = !
invocable_on_elements< T >&& detail::is_invocable_on_elements< T, false, Args... >::value;
template < typename T, typename... Args >
concept invocable_on_const_elements_and = !
invocable_on_const_elements< T >&& detail::is_invocable_on_elements< T, true, Args... >::value;

namespace detail
{
template < typename R, typename F, bool is_const, typename... Args >
struct is_invocable_r_on_elements
{
    template < ElementTypes ET, el_o_t EO >
    struct invocable_on_element :
        std::conditional_t<
            std::is_invocable_r_v< R,
                                   F,
                                   std::conditional_t< is_const, const Element< ET, EO >&, Element< ET, EO >& >,
                                   Args... >,
            std::true_type,
            std::false_type >
    {};

    using invocability_set = parametrize_type_over_element_types_and_orders_t< TypePack, invocable_on_element >;

    template < typename >
    struct check_all;
    template < typename... T >
    struct check_all< TypePack< T... > >
    {
        static constexpr bool value = (T::value && ...);
    };

    static constexpr bool value = check_all< invocability_set >::value;
};
} // namespace detail

template < typename T, typename R >
concept invocable_on_elements_r = detail::is_invocable_r_on_elements< R, T, false >::value;
template < typename T, typename R >
concept invocable_on_const_elements_r = detail::is_invocable_r_on_elements< R, T, true >::value;
template < typename T, typename R, typename... Args >
concept invocable_on_elements_r_and = !
invocable_on_elements_r< T, R >&& detail::is_invocable_r_on_elements< R, T, false, Args... >::value;
template < typename T, typename R, typename... Args >
concept invocable_on_const_elements_r_and = !
invocable_on_const_elements_r< T, R >&& detail::is_invocable_r_on_elements< R, T, true, Args... >::value;
} // namespace lstr
#endif // L3STER_MESH_ALIASES_HPP
