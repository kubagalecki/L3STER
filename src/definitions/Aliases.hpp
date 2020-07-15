#ifndef L3STER_DEFINITIONS_ALIASES_HPP
#define L3STER_DEFINITIONS_ALIASES_HPP

#include <variant>

namespace lstr::mesh
{
template < template < typename... > typename T,
           template < mesh::ElementTypes, types::el_o_t >
           typename U >
using parametrize_over_element_types_and_orders_t =
    util::meta::cartesian_product_t< T, U, ElementTypesArray, ElementOrdersArray >;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element;
using element_variant_t =
    util::meta::cartesian_product_t< std::variant, Element, ElementTypesArray, ElementOrdersArray >;
} // namespace lstr::mesh

#endif // L3STER_DEFINITIONS_ALIASES_HPP
