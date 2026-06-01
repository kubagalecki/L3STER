#ifndef L3STER_MESH_ELEMENTMETA_HPP
#define L3STER_MESH_ELEMENTMETA_HPP

#include "l3ster/common/Typedefs.h"
#include "l3ster/mesh/ElementType.hpp"
#include "l3ster/util/Algorithm.hpp"

#include <variant>

namespace lstr::mesh
{
template < ElementType ET, el_o_t EO >
struct Element;

namespace detail
{
template < el_o_t... orders >
using ElementTypesPack =
    util::CartesianProductApply< util::ValuePack, util::TypePack, element_types, std::array{orders...} >;

template < template < typename... > typename Outer, template < ElementType, el_o_t > typename Inner, el_o_t... orders >
class CalculateAppliedType
{
    template < typename >
    struct SubInner
    {};
    template < typename >
    struct SubOuter
    {};

    template < ElementType ET, el_o_t EO >
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

template < auto init, template < ElementType, el_o_t > typename Transform, typename Reduce, el_o_t... orders >
class ReductionImpl
{
    template < typename >
    struct TransResult
    {};
    template < typename >
    struct ReduceResult
    {};

    template < ElementType ET, el_o_t EO >
    struct TransResult< util::ValuePack< ET, EO > >
    {
        static constexpr auto value = Transform< ET, EO >::value;
    };
    template < typename... Ts >
    struct ReduceResult< util::TypePack< Ts... > >
    {
        static constexpr auto value = std::ranges::fold_left(std::array{TransResult< Ts >::value...}, init, Reduce{});
    };

public:
    static constexpr auto value = ReduceResult< ElementTypesPack< orders... > >::value;
};
} // namespace detail

template < template < typename... > typename Outer, template < ElementType, el_o_t > typename Inner, el_o_t... orders >
using parametrize_type_over_element_types_and_orders_t = detail::CalculateAppliedType< Outer, Inner, orders... >::type;

template < bool is_const, ElementType ET, el_o_t EO >
using cond_const_elref_t = std::conditional_t< is_const, const Element< ET, EO >&, Element< ET, EO >& >;

template < ElementType ET, el_o_t EO >
class BoundaryElementView;

template < auto init, template < ElementType, el_o_t > typename Transform, typename Reduce, el_o_t... orders >
inline constexpr auto meta_transform_reduce = detail::ReductionImpl< init, Transform, Reduce, orders... >::value;

template < el_o_t... orders >
class ElementDeductionHelper
{
    template < bool is_const, typename F, typename... Args >
    struct InvokeHelper
    {
        template < ElementType ET, el_o_t EO >
        struct Helper : std::is_invocable< F, cond_const_elref_t< is_const, ET, EO >, Args... >
        {};
        static constexpr bool value = meta_transform_reduce< true, Helper, std::logical_and<>, orders... >;
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
        static constexpr bool value = meta_transform_reduce< true, Helper, std::logical_and<>, orders... >;
    };
    template < typename F, typename... Args >
    struct BoundaryInvokeHelper
    {
        template < ElementType ET, el_o_t EO >
        struct Helper : std::is_invocable< F, BoundaryElementView< ET, EO >, Args... >
        {};
        static constexpr bool value = meta_transform_reduce< true, Helper, std::logical_and<>, orders... >;
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
        static constexpr bool value = meta_transform_reduce< true, Helper, std::logical_and<>, orders... >;
    };

public:
    template < typename R, typename F, typename... Args >
    static constexpr bool invocable_on_const_elements_return = InvokeReturnHelper< true, R, F, Args... >::value;
    template < typename R, typename F, typename... Args >
    static constexpr bool invocable_on_boundary_views_return = BoundaryInvokeReturnHelper< R, F, Args... >::value;
};
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENTMETA_HPP
