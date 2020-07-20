#ifndef L3STER_DEFINITIONS_ALIASES_HPP
#define L3STER_DEFINITIONS_ALIASES_HPP

#include <functional>
#include <variant>

namespace lstr::mesh
{
template < template < typename... > typename T,
           template < mesh::ElementTypes, types::el_o_t >
           typename U >
using parametrize_type_over_element_types_and_orders_t =
    util::meta::cartesian_product_t< T, U, ElementTypesArray, ElementOrdersArray >;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element;

using element_variant_t = parametrize_type_over_element_types_and_orders_t< std::variant, Element >;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
using element_ref_t = std::reference_wrapper< Element< ELTYPE, ELORDER > >;

using element_ref_variant_t =
    parametrize_type_over_element_types_and_orders_t< std::variant, element_ref_t >;

template < typename F >
struct is_invocable_on_all_elements
{
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    using is_invocable_on_element = std::is_invocable< F, Element< ELTYPE, ELORDER >& >;

    static constexpr bool value =
        parametrize_type_over_element_types_and_orders_t< util::meta::and_pack,
                                                          is_invocable_on_element >::value;
};

template < typename F >
inline constexpr bool is_invocable_on_all_elements_v = is_invocable_on_all_elements< F >::value;

template < typename F, typename R >
struct is_invocable_r_on_all_elements
{
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    using is_invocable_r_on_element = std::is_invocable_r< R, F, Element< ELTYPE, ELORDER >& >;

    static constexpr bool value =
        parametrize_type_over_element_types_and_orders_t< util::meta::and_pack,
                                                          is_invocable_r_on_element >::value;
};

template < typename F, typename R >
inline constexpr bool is_invocable_r_on_all_elements_v =
    is_invocable_r_on_all_elements< F, R >::value;

} // namespace lstr::mesh

#endif // L3STER_DEFINITIONS_ALIASES_HPP
