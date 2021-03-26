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
template < ElementTypes ELTYPE, el_o_t ELORDER >
class Element
{
public:
    using node_array_t = std::array< n_id_t, ElementTraits< Element< ELTYPE, ELORDER > >::nodes_per_element >;

    explicit Element(const node_array_t& nodes_) : nodes{nodes_} {}

    [[nodiscard]] const node_array_t& getNodes() const noexcept { return nodes; }
    [[nodiscard]] node_array_t&       getNodes() noexcept { return nodes; }

private:
    node_array_t                                                      nodes;
    typename ElementTraits< Element< ELTYPE, ELORDER > >::ElementData data{};
};
} // namespace lstr
#endif // L3STER_MESH_ELEMENT_HPP
