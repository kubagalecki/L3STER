#ifndef L3STER_MESH_ELEMENTMETA_HPP
#define L3STER_MESH_ELEMENTMETA_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/mesh/ElementType.hpp"
#include "l3ster/util/Algorithm.hpp"

#include <array>
#include <functional>
#include <variant>

namespace lstr::mesh
{
template < ElementType ET, el_o_t EO >
class Element;

namespace detail
{
template < el_o_t... orders >
using ElementTypesPack =
    util::CartesianProductApply< util::ValuePack, util::TypePack, element_types, std::array{orders...} >;

template < template < typename... > typename Outer, template < ElementType, el_o_t > typename Inner, el_o_t... orders >
class CalculateAppliedType
{
    template < typename >
    struct SubInner;
    template < typename T >
    struct SubOuter;

    template < mesh::ElementType ET, el_o_t EO >
    struct SubInner< util::ValuePack< ET, EO > >
    {
        using type = Inner< ET, EO >;
    };
    template < typename... Ts >
    struct SubOuter< util::TypePack< Ts... > >
    {
        using type = Outer< typename SubInner< Ts >::type... >;
    };

public:
    using type = SubOuter< ElementTypesPack< orders... > >::type;
};
} // namespace detail

template < template < typename... > typename Outer, template < ElementType, el_o_t > typename Inner, el_o_t... orders >
using parametrize_type_over_element_types_and_orders_t = detail::CalculateAppliedType< Outer, Inner, orders... >::type;

template < ElementType ET, el_o_t EO >
using element_cptr_t = const Element< ET, EO >*;
template < el_o_t... orders >
using element_cptr_variant_t =
    parametrize_type_over_element_types_and_orders_t< std::variant, element_cptr_t, orders... >;
template < ElementType ET, el_o_t EO >
class BoundaryElementView;

template < bool is_const, ElementType ET, el_o_t EO >
using cond_const_elref_t = std::conditional_t< is_const, const Element< ET, EO >&, Element< ET, EO >& >;

template < el_o_t... orders >
class ElementDeductionHelper
{
    static_assert(sizeof...(orders) > 0);

public:
    template < template < ElementType, el_o_t > typename Condition >
    static constexpr bool assert_all_elements =
        parametrize_type_over_element_types_and_orders_t< std::conjunction, Condition, orders... >::value;

    template < template < ElementType, el_o_t > typename Condition >
    static constexpr bool assert_any_element =
        parametrize_type_over_element_types_and_orders_t< std::disjunction, Condition, orders... >::value;

private:
    template < bool is_const, typename F, typename... Args >
    struct InvokeHelper
    {
        template < ElementType ET, el_o_t EO >
        struct Helper : std::is_invocable< F, cond_const_elref_t< is_const, ET, EO >, Args... >
        {};
        static constexpr bool value = assert_all_elements< Helper >;
    };
    template < bool is_const, typename R, typename F, typename... Args >
    struct InvokeReturnHelper
    {
        template < ElementType ET, el_o_t EO >
        struct Helper
        {
            using element_t             = cond_const_elref_t< is_const, ET, EO >;
            static constexpr bool value = ReturnInvocable_c< F, R, element_t, Args... >;
        };
        static constexpr bool value = assert_all_elements< Helper >;
    };
    template < typename F, typename... Args >
    struct BoundaryInvokeHelper
    {
        template < ElementType ET, el_o_t EO >
        struct Helper : std::is_invocable< F, BoundaryElementView< ET, EO >, Args... >
        {};
        static constexpr bool value = assert_all_elements< Helper >;
    };
    template < typename R, typename F, typename... Args >
    struct BoundaryInvokeReturnHelper
    {
        template < ElementType ET, el_o_t EO >
        struct Helper
        {
            using bview_t               = BoundaryElementView< ET, EO >;
            static constexpr bool value = ReturnInvocable_c< F, R, bview_t, Args... >;
        };
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
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENTMETA_HPP
