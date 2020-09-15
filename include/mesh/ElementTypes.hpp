// Enumeration containing element types

#ifndef L3STER_INCGUARD_MESH_ELEMENTTYPES_H
#define L3STER_INCGUARD_MESH_ELEMENTTYPES_H

#include <array>
#include <utility>

namespace lstr::mesh
{
enum class ElementTypes
{
    Quad, // Quadrilateral elements (geometrically linear)
    Line, // line segment (2 nodes, geometrically linear)

    //////////////////////////////////////////////////////////////////////////////
    //////////          NEW ELEMENT TYPES BEFORE THIS LINE              //////////
    //////////////////////////////////////////////////////////////////////////////
    Count // value for tracking number of Element Types
};

namespace helpers
{
// Constexpr array of element types
template < size_t... Ints >
constexpr auto makeElementTypeArray(std::index_sequence< Ints... >)
{
    return std::array< ElementTypes, sizeof...(Ints) >{static_cast< ElementTypes >(Ints)...};
}

} // namespace helpers

// Array containing all defined element types
struct ElementTypesArray
{
    static constexpr auto values = helpers::makeElementTypeArray(
        std::make_index_sequence< static_cast< size_t >(ElementTypes::Count) >{});
};

// Array containing all possible element orders.
struct ElementOrdersArray
{
    static constexpr std::array values = allowed_orders;
};
} // namespace lstr::mesh

#endif // end include guard
