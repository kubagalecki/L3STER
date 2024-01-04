#ifndef L3STER_MESH_ELEMENTTYPES_HPP
#define L3STER_MESH_ELEMENTTYPES_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <utility>

namespace lstr::mesh
{
enum struct ElementType
{
    Hex  = 0,
    Quad = 1,
    Line = 2,

    // Update Count when adding new element types
    Count = 3
};

// Array containing all defined element types
inline constexpr auto element_types = std::invoke([] {
    auto retval = std::array< ElementType, static_cast< size_t >(ElementType::Count) >{};
    std::ranges::generate(retval, [i = 0]() mutable { return static_cast< ElementType >(i++); });
    return retval;
});
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENTTYPES_HPP
