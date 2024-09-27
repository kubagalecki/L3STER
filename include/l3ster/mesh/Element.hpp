#ifndef L3STER_MESH_ELEMENT_HPP
#define L3STER_MESH_ELEMENT_HPP

#include "l3ster/mesh/ElementData.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lstr::mesh
{
template < ElementType T, el_o_t O >
class Element
{
public:
    static constexpr ElementType type  = T;
    static constexpr el_o_t      order = O;

    static constexpr size_t n_nodes    = ElementTraits< Element< T, O > >::nodes_per_element;
    static constexpr auto   native_dim = ElementTraits< Element< T, O > >::native_dim;
    using node_array_t                 = std::array< n_id_t, n_nodes >;

    constexpr Element() = default;
    constexpr Element(const node_array_t& nodes_, const ElementData< T, O >& data_, el_id_t id_) noexcept
        : nodes{nodes_}, data{data_}, id{id_}
    {}

    [[nodiscard]] constexpr auto getNodes() const -> const node_array_t& { return nodes; }
    [[nodiscard]] constexpr auto getNodes() -> node_array_t& { return nodes; }
    [[nodiscard]] constexpr auto getData() const -> const ElementData< T, O >& { return data; }
    [[nodiscard]] constexpr auto getId() const -> el_id_t { return id; }

    friend constexpr bool operator==(const Element&, const Element&) = default;

private:
    node_array_t        nodes;
    ElementData< T, O > data;
    el_id_t             id;
};

template < ElementType T, el_o_t O >
constexpr auto getInternalNodes(const Element< T, O >& element)
{
    return ElementTraits< Element< T, O > >::internal_node_inds |
           std::views::transform([&](size_t i) { return element.getNodes()[i]; });
}

template < ElementType T, el_o_t O >
constexpr auto getBoundaryNodes(const Element< T, O >& element)
{
    return ElementTraits< Element< T, O > >::boundary_node_inds |
           std::views::transform([&](size_t i) { return element.getNodes()[i]; });
}
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENT_HPP
