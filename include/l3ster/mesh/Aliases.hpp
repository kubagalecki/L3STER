#ifndef L3STER_MESH_ALIASES_HPP
#define L3STER_MESH_ALIASES_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/mesh/ElementTypes.hpp"
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
template < ElementTypes ET, el_o_t EO >
class BoundaryElementView;

template < ElementTypes TYPE, el_o_t ORDER >
using type_order_set          = ValuePack< TYPE, ORDER >;
using type_order_combinations = parametrize_type_over_element_types_and_orders_t< TypePack, type_order_set >;

namespace detail
{
template < bool is_const, ElementTypes ET, el_o_t EO >
using cond_const_elref_t = std::conditional_t< is_const, const Element< ET, EO >&, Element< ET, EO >& >;

template < template < ElementTypes, el_o_t > typename Condition >
inline constexpr bool assert_all_elements =
    parametrize_type_over_element_types_and_orders_t< std::conjunction, Condition >::value;
template < template < ElementTypes, el_o_t > typename Condition >
inline constexpr bool assert_any_element =
    parametrize_type_over_element_types_and_orders_t< std::disjunction, Condition >::value;

template < typename F, bool is_const, typename... Args >
struct InvocableOnElementsHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelper : std::is_invocable< F, cond_const_elref_t< is_const, ET, EO >, Args... >
    {};

    static constexpr bool value = assert_all_elements< DeductionHelper >;
};
template < typename F, bool is_const, typename... Args >
inline constexpr bool is_invocable_on_elements = InvocableOnElementsHelper< F, is_const, Args... >::value;

template < typename R, typename F, bool is_const, typename... Args >
struct InvocableOnElementsRetHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelper : std::is_invocable_r< R, F, cond_const_elref_t< is_const, ET, EO >, Args... >
    {};

    static constexpr bool value = assert_all_elements< DeductionHelper >;
};
template < typename R, typename F, bool is_const, typename... Args >
inline constexpr bool is_invocable_r_on_elements = InvocableOnElementsRetHelper< R, F, is_const, Args... >::value;

template < typename F >
struct InvocableOnBoundaryElementViewsHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelper : std::is_invocable< F, const BoundaryElementView< ET, EO > >
    {};

    static constexpr bool value = assert_all_elements< DeductionHelper >;
};
template < typename R, typename F >
struct InvocableOnBoundaryElementViewsRetHelper
{
    template < ElementTypes ET, el_o_t EO >
    struct DeductionHelper : std::is_invocable_r< R, F, const BoundaryElementView< ET, EO > >
    {};

    static constexpr bool value = assert_all_elements< DeductionHelper >;
};
template < typename F >
inline constexpr bool is_invocable_on_boundary_element_views = InvocableOnBoundaryElementViewsHelper< F >::value;
template < typename R, typename F >
inline constexpr bool is_invocable_r_on_boundary_element_views =
    InvocableOnBoundaryElementViewsRetHelper< R, F >::value;
} // namespace detail

template < typename T >
concept invocable_on_elements = detail::is_invocable_on_elements< T, false >;
template < typename T >
concept invocable_on_const_elements = detail::is_invocable_on_elements< T, true >;
template < typename T, typename... Args >
concept invocable_on_elements_and = not
invocable_on_elements< T >&& detail::is_invocable_on_elements< T, false, Args... >;
template < typename T, typename... Args >
concept invocable_on_const_elements_and = not
invocable_on_const_elements< T >&& detail::is_invocable_on_elements< T, true, Args... >;

template < typename T, typename R >
concept invocable_on_elements_r = detail::is_invocable_r_on_elements< R, T, false >;
template < typename T, typename R >
concept invocable_on_const_elements_r = detail::is_invocable_r_on_elements< R, T, true >;
template < typename T, typename R, typename... Args >
concept invocable_on_elements_r_and = not
invocable_on_elements_r< T, R >&& detail::is_invocable_r_on_elements< R, T, false, Args... >;
template < typename T, typename R, typename... Args >
concept invocable_on_const_elements_r_and = not
invocable_on_const_elements_r< T, R >&& detail::is_invocable_r_on_elements< R, T, true, Args... >;

template < typename T >
concept invocable_on_boundary_element_views = detail::is_invocable_on_boundary_element_views< T >;
template < typename T, typename R >
concept invocable_on_boundary_element_views_r = detail::is_invocable_r_on_boundary_element_views< R, T >;
} // namespace lstr
#endif // L3STER_MESH_ALIASES_HPP
