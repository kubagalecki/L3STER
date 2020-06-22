// Data type representing a physical element

#ifndef L3STER_INCGUARD_MESH_ELEMENT_HPP
#define L3STER_INCGUARD_MESH_ELEMENT_HPP

#include "definitions/Typedefs.h"
#include "mesh/ElementTraits.hpp"
#include "utility/Meta.hpp"

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
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element
{
public:
    using node_array_t =
        std::array< types::n_id_t, ElementTraits< Element< ELTYPE, ELORDER > >::nodes_per_element >;
    using node_array_ref_t      = node_array_t&;
    using node_array_constref_t = const node_array_t&;

    Element(const Element&)     = default;
    Element(Element&&) noexcept = default;
    Element& operator=(const Element&) = default;
    Element& operator=(Element&&) noexcept = default;

    explicit Element(const node_array_t& n) : nodes(n) {}

private:
    node_array_t nodes = node_array_t{};
    // std::array< size_t, n_nodes > node_order;

    // element data
    // typename ElementTraits< Element< ELTYPE, ELORDER > >::ElementData data;
};
} // namespace lstr::mesh

#endif // end include guard
