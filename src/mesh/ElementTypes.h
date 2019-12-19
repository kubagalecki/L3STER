// Enumeration containing element types

#ifndef L3STER_INCGUARD_MESH_ELEMENTTYPES_H
#define L3STER_INCGUARD_MESH_ELEMENTTYPES_H

#include "typedefs/Types.h"

#include <array>
#include <utility>

namespace lstr
{
namespace mesh
{
enum class ElementTypes
{
    Quad,                   // Quadrilateral elements (geometrically linear)

    // NEW ELEMENT TYPES BEFORE THIS LINE //
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

constexpr auto element_type_array = helpers::makeElementTypeArray(
    std::make_index_sequence<static_cast<size_t>(ElementTypes::Count)>{});

}           // namespace mesh
}           // namespace lstr

#endif      // end include guard
