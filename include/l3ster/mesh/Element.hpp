#ifndef L3STER_MESH_ELEMENT_HPP
#define L3STER_MESH_ELEMENT_HPP

#include "ElementData.hpp"

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
    static constexpr ElementTypes type  = T;
    static constexpr el_o_t       order = O;

    static constexpr size_t n_nodes = ElementTraits< Element< T, O > >::nodes_per_element;
    using node_array_t              = std::array< n_id_t, n_nodes >;
    using element_data_t            = ElementData< T, O >::ElementData;

    Element(const node_array_t& nodes_, const element_data_t& data_, el_id_t id_) noexcept
        : nodes{nodes_}, data{data_}, id{id_}
    {}

    [[nodiscard]] const node_array_t&   getNodes() const noexcept { return nodes; }
    [[nodiscard]] node_array_t&         getNodes() noexcept { return nodes; }
    [[nodiscard]] const element_data_t& getData() const noexcept { return data; }
    [[nodiscard]] el_id_t               getId() const noexcept { return id; }

private:
    node_array_t   nodes;
    element_data_t data;
    el_id_t        id;
};
} // namespace lstr
#endif // L3STER_MESH_ELEMENT_HPP
