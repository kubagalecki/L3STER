// Data type representing a physical element

#ifndef L3STER_MESH_ELEMENT_HPP
#define L3STER_MESH_ELEMENT_HPP

#include "mesh/ElementTraits.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lstr
{
template < ElementTypes T, el_o_t O >
class Element
{
public:
    using node_array_t   = std::array< n_id_t, ElementTraits< Element< T, O > >::nodes_per_element >;
    using element_data_t = ElementTraits< Element< T, O > >::ElementData;

    constexpr Element(const node_array_t& nodes_, el_id_t id_) noexcept : nodes{nodes_}, id{id_} {}

    [[nodiscard]] const node_array_t& getNodes() const noexcept { return nodes; }
    [[nodiscard]] node_array_t&       getNodes() noexcept { return nodes; }
    [[nodiscard]] el_id_t             getId() const noexcept { return id; }

private:
    node_array_t   nodes;
    element_data_t data{};
    el_id_t        id;
};
} // namespace lstr
#endif // L3STER_MESH_ELEMENT_HPP
