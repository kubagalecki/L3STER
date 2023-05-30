#ifndef L3STER_MESH_ALIASES_HPP
#define L3STER_MESH_ALIASES_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/mesh/ElementTypes.hpp"
#include "l3ster/util/Algorithm.hpp"

#include <array>
#include <functional>
#include <variant>

namespace lstr
{
namespace detail
{
template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
inline constexpr auto element_orders_array = util::sortedUniqueElements< std::array{orders...} >();
}
template < template < typename... > typename T, template < ElementTypes, el_o_t > typename U, el_o_t... orders >
using parametrize_type_over_element_types_and_orders_t =
    util::cart_prod_t< U, T, element_types, detail::element_orders_array< orders... > >;

template < ElementTypes ET, el_o_t EO >
class Element;
template < ElementTypes ET, el_o_t EO >
using element_ptr_t = Element< ET, EO >*;
template < ElementTypes ET, el_o_t EO >
using element_cptr_t = const Element< ET, EO >*;
template < el_o_t... orders >
using element_ptr_variant_t =
    parametrize_type_over_element_types_and_orders_t< std::variant, element_ptr_t, orders... >;
template < el_o_t... orders >
using element_cptr_variant_t =
    parametrize_type_over_element_types_and_orders_t< std::variant, element_cptr_t, orders... >;
template < ElementTypes ET, el_o_t EO >
class BoundaryElementView;

template < ElementTypes ET, el_o_t EO >
using type_order_set = util::ValuePack< ET, EO >;
template < el_o_t... orders >
using type_order_combinations =
    parametrize_type_over_element_types_and_orders_t< util::TypePack, type_order_set, orders... >;

namespace detail
{
template < bool is_const, ElementTypes ET, el_o_t EO >
using cond_const_elref_t = std::conditional_t< is_const, const Element< ET, EO >&, Element< ET, EO >& >;

template < el_o_t... orders >
    requires(sizeof...(orders) > 0)
class ElementDeductionHelper
{
public:
    template < template < ElementTypes, el_o_t > typename Condition >
    static constexpr bool assert_all_elements =
        parametrize_type_over_element_types_and_orders_t< std::conjunction, Condition, orders... >::value;

    template < template < ElementTypes, el_o_t > typename Condition >
    static constexpr bool assert_any_element =
        parametrize_type_over_element_types_and_orders_t< std::disjunction, Condition, orders... >::value;

private:
    template < bool is_const, typename F, typename... Args >
    struct InvokeHelper
    {
        template < ElementTypes ET, el_o_t EO >
        struct Helper : std::is_invocable< F, cond_const_elref_t< is_const, ET, EO >, Args... >
        {};
        static constexpr bool value = assert_all_elements< Helper >;
    };
    template < bool is_const, typename R, typename F, typename... Args >
    struct InvokeReturnHelper
    {
        template < ElementTypes ET, el_o_t EO >
        struct Helper : std::is_invocable_r< R, F, cond_const_elref_t< is_const, ET, EO >, Args... >
        {};
        static constexpr bool value = assert_all_elements< Helper >;
    };
    template < typename F, typename... Args >
    struct BoundaryInvokeHelper
    {
        template < ElementTypes ET, el_o_t EO >
        struct Helper : std::is_invocable< F, BoundaryElementView< ET, EO >, Args... >
        {};
        static constexpr bool value = assert_all_elements< Helper >;
    };
    template < typename R, typename F, typename... Args >
    struct BoundaryInvokeReturnHelper
    {
        template < ElementTypes ET, el_o_t EO >
        struct Helper : std::is_invocable_r< R, F, BoundaryElementView< ET, EO >, Args... >
        {};
        static constexpr bool value = assert_all_elements< Helper >;
    };

public:
    template < typename F, typename... Args >
    static constexpr bool invocable_on_elements = InvokeHelper< false, F, Args... >::value;
    template < typename F, typename... Args >
    static constexpr bool invocable_on_const_elements = InvokeHelper< true, F, Args... >::value;
    template < typename R, typename F, typename... Args >
    static constexpr bool invocable_on_elements_return = InvokeReturnHelper< false, R, F, Args... >::value;
    template < typename R, typename F, typename... Args >
    static constexpr bool invocable_on_const_elements_return = InvokeReturnHelper< true, R, F, Args... >::value;
    template < typename F, typename... Args >
    static constexpr bool invocable_on_boundary_views = BoundaryInvokeHelper< F, Args... >::value;
    template < typename R, typename F, typename... Args >
    static constexpr bool invocable_on_boundary_views_return = BoundaryInvokeReturnHelper< R, F, Args... >::value;
};
} // namespace detail
} // namespace lstr
#endif // L3STER_MESH_ALIASES_HPP
