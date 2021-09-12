#ifndef L3STER_MESH_ELEMENTDATA_HPP
#define L3STER_MESH_ELEMENTDATA_HPP

#include "mesh/ElementTraits.hpp"
#include "mesh/Point.hpp"

namespace lstr
{
template < ElementTypes T, el_o_t O >
struct ElementData;

template < ElementTypes T, el_o_t O >
requires(T == ElementTypes::Line or T == ElementTypes::Quad or T == ElementTypes::Hex) struct ElementData< T, O >
{
    static constexpr auto n_verts = ElementTraits< Element< T, 1 > >::nodes_per_element;
    using vertex_array_t          = std::array< Point< 3 >, n_verts >;

    ElementData() = default;
    ElementData(const vertex_array_t& vertices_) : vertices{vertices_} {} // NOLINT implicit conversion intended
    template < el_o_t O_ >                                                // NOLINTNEXTLINE implicit conversion intended
    requires(O != O_) ElementData(const ElementData< T, O_ >& d) : vertices{d.vertices} {}

    vertex_array_t vertices;
};
} // namespace lstr
#endif // L3STER_MESH_ELEMENTDATA_HPP
