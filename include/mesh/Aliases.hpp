#ifndef L3STER_MESH_ALIASES_HPP
#define L3STER_MESH_ALIASES_HPP

#include "defs/Typedefs.h"
#include "mesh/ElementTypes.hpp"
#include "util/Meta.hpp"

#include <functional>
#include <variant>

namespace lstr::mesh
{
template < template < typename... > typename T, template < mesh::ElementTypes, types::el_o_t > typename U >
using parametrize_type_over_element_types_and_orders_t =
    util::meta::parametrize_over_combinations_t< U, T, element_types, element_orders >;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element;

using element_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, Element >;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
using element_ref_t = std::reference_wrapper< Element< ELTYPE, ELORDER > >;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
using element_cref_t = std::reference_wrapper< const Element< ELTYPE, ELORDER > >;

using element_ref_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, element_ref_t >;

using element_cref_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, element_cref_t >;

namespace detail
{
template < typename F, bool CONSTNESS >
struct is_invocable_on_elements
{
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    struct invocable_on_element :
        std::conditional_t<
            std::is_invocable_v<
                F,
                std::conditional_t< CONSTNESS, const Element< ELTYPE, ELORDER >&, Element< ELTYPE, ELORDER >& > >,
            std::true_type,
            std::false_type >
    {};

    using invocability_tuple = parametrize_type_over_element_types_and_orders_t< std::tuple, invocable_on_element >;

    template < typename >
    struct check_all;
    template < typename... T >
    struct check_all< std::tuple< T... > >
    {
        static constexpr bool value = (T::value && ...);
    };

    static constexpr bool value = check_all< invocability_tuple >::value;
};
} // namespace detail

template < typename T >
concept invocable_on_elements = detail::is_invocable_on_elements< T, false >::value;
template < typename T >
concept invocable_on_const_elements = detail::is_invocable_on_elements< T, true >::value;

namespace detail
{
template < typename R, typename F, bool CONSTNESS >
struct is_invocable_r_on_elements
{
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    struct invocable_on_element :
        std::conditional_t<
            std::is_invocable_r_v<
                R,
                F,
                std::conditional_t< CONSTNESS, const Element< ELTYPE, ELORDER >&, Element< ELTYPE, ELORDER >& > >,
            std::true_type,
            std::false_type >
    {};

    using invocability_tuple = parametrize_type_over_element_types_and_orders_t< std::tuple, invocable_on_element >;

    template < typename >
    struct check_all;
    template < typename... T >
    struct check_all< std::tuple< T... > >
    {
        static constexpr bool value = (T::value && ...);
    };

    static constexpr bool value = check_all< invocability_tuple >::value;
};
} // namespace detail

template < typename T, typename R >
concept invocable_on_elements_r = detail::is_invocable_r_on_elements< R, T, false >::value;
template < typename T, typename R >
concept invocable_on_const_elements_r = detail::is_invocable_r_on_elements< R, T, true >::value;

} // namespace lstr::mesh

#endif // L3STER_MESH_ALIASES_HPP
