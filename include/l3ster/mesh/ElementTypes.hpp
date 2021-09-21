#ifndef L3STER_MESH_ELEMENTTYPES_H
#define L3STER_MESH_ELEMENTTYPES_H

#include "Constants.hpp"

#include <array>
#include <utility>

namespace lstr
{
enum class ElementTypes : std::uint_fast8_t
{
    Hex,  // Hexahedral elements (geometrically linear)
    Quad, // Quadrilateral elements (geometrically linear)
    Line, // line segment (2 nodes, geometrically linear)
    // !!! NEW ELEMENT TYPES BEFORE THIS LINE !!!
    Count // value for tracking number of Element Types
};

// Array containing all defined element types
inline constexpr auto element_types = []< size_t... I >(std::index_sequence< I... >)
{
    return std::array{static_cast< ElementTypes >(I)...};
}
(std::make_index_sequence< static_cast< size_t >(ElementTypes::Count) >{});

} // namespace lstr
#endif // L3STER_MESH_ELEMENTTYPES_H
