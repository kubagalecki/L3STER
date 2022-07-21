#ifndef L3STER_MESH_ELEMENTTYPES_H
#define L3STER_MESH_ELEMENTTYPES_H

#include "l3ster/mesh/Constants.hpp"

#include <array>
#include <utility>

#define L3STER_SIZED_ENUM(name, type, ...)                                                                             \
    enum class name : type                                                                                             \
    {                                                                                                                  \
        __VA_ARGS__,                                                                                                   \
        Count                                                                                                          \
    }

namespace lstr
{
L3STER_SIZED_ENUM(ElementTypes, std::uint8_t, Hex, Quad, Line);

// Array containing all defined element types
inline constexpr auto element_types = []< size_t... I >(std::index_sequence< I... >)
{
    return std::array{static_cast< ElementTypes >(I)...};
}
(std::make_index_sequence< static_cast< size_t >(ElementTypes::Count) >{});
} // namespace lstr
#endif // L3STER_MESH_ELEMENTTYPES_H
