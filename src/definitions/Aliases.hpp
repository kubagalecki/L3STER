#ifndef L3STER_INCGUARD_DEFINITIONS_ALIASES_HPP
#define L3STER_INCGUARD_DEFINITIONS_ALIASES_HPP

namespace lstr::mesh
{
// Define alias for easy templating over all possible element/order combinations
template < template < typename... > typename T,
           template < mesh::ElementTypes, types::el_o_t >
           typename U >
using parametrize_over_element_types_and_orders_t =
    typename util::meta::cartesian_product_t< T, U, ElementTypesArray, ElementOrdersArray >::type;
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_DEFINITIONS_ALIASES_HPP
