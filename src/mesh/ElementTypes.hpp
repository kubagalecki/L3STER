// Enumeration containing element types

#ifndef L3STER_INCGUARD_MESH_ELEMENTTYPES_H
#define L3STER_INCGUARD_MESH_ELEMENTTYPES_H

#include "typedefs/Types.h"

#include <array>
#include <utility>

namespace lstr::mesh
{
enum class ElementTypes
{
    Quad,                   // Quadrilateral elements (geometrically linear)

    //////////////////////////////////////////////////////////////////////////////
    //////////          NEW ELEMENT TYPES BEFORE THIS LINE              //////////
    //////////////////////////////////////////////////////////////////////////////

    Count                   // value for tracking number of Element Types
};

namespace helpers
{

// Use meta-programming to create constexpr array of element types
template<size_t ... Ints>
constexpr auto makeElementTypeArray(std::index_sequence<Ints ...>)
{
    return std::array<ElementTypes, sizeof...(Ints)> {static_cast<ElementTypes>(Ints) ...};
}

}           // namespace helpers

// Array containing all defined element types
struct ElementTypesArray
{

    static constexpr auto values = helpers::makeElementTypeArray(
    std::make_index_sequence< static_cast<size_t>(ElementTypes::Count) > {});

};

// Array containing all possible element orders.
struct ElementOrdersArray
{

    static constexpr std::array<types::el_o_t, 10> values = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10};

};

}           // namespace lstr::mesh

#endif      // end include guard
