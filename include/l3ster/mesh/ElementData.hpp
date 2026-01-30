#ifndef L3STER_MESH_ELEMENTDATA_HPP
#define L3STER_MESH_ELEMENTDATA_HPP

#include "l3ster/common/Structs.hpp"
#include "l3ster/mesh/ElementTraits.hpp"

namespace lstr::mesh
{
template < ElementType T, el_o_t O >
struct ElementData;

template < ElementType T, el_o_t O >
    requires(T == ElementType::Line or T == ElementType::Quad or T == ElementType::Hex)
struct ElementData< T, O >
{
    static constexpr auto n_verts = ElementTraits< Element< T, 1 > >::nodes_per_element;
    using vertex_array_t          = std::array< Point< 3 >, n_verts >;

    constexpr ElementData() = default;
    constexpr ElementData(const vertex_array_t& vertices_) : vertices{vertices_} {}
    template < el_o_t O_ >
    constexpr ElementData(const ElementData< T, O_ >& d)
        requires(O != O_)
        : vertices{d.vertices}
    {}

    auto getEigenMap() const
    {
        using matrix_t = Eigen::Matrix< val_t, 3, n_verts >;
        return Eigen::Map< const matrix_t >{vertices.front().coords.data()};
    }

    friend constexpr bool operator==(const ElementData&, const ElementData&) = default;

    vertex_array_t vertices;
};
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENTDATA_HPP
