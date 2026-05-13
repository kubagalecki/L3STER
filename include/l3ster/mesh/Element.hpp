#ifndef L3STER_MESH_ELEMENT_HPP
#define L3STER_MESH_ELEMENT_HPP

#include "l3ster/mesh/ElementData.hpp"

#include <array>

namespace lstr::mesh
{
template < ElementType T, el_o_t O >
struct Element
{
    static constexpr ElementType type       = T;
    static constexpr el_o_t      order      = O;
    static constexpr size_t      n_nodes    = ElementTraits< Element >::nodes_per_element;
    static constexpr auto        native_dim = ElementTraits< Element >::native_dim;
    using node_array_t                      = std::array< n_id_t, n_nodes >;

    friend constexpr bool operator==(const Element&, const Element&) = default;

    node_array_t        nodes;
    ElementData< T, O > data;
    el_id_t             id;
};

template < ElementType T, el_o_t O >
constexpr auto getInternalNodes(const Element< T, O >& element)
{
    return ElementTraits< Element< T, O > >::internal_node_inds |
           std::views::transform([&](size_t i) { return element.nodes[i]; });
}

template < ElementType T, el_o_t O >
constexpr auto getBoundaryNodes(const Element< T, O >& element)
{
    return ElementTraits< Element< T, O > >::boundary_node_inds |
           std::views::transform([&](size_t i) { return element.nodes[i]; });
}
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENT_HPP
