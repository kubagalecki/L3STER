#ifndef L3STER_MESH_ELEMENTTYPES_HPP
#define L3STER_MESH_ELEMENTTYPES_HPP

#include <algorithm>
#include <array>
#include <functional>
#include <utility>

namespace lstr
{
enum struct ElementTypes
{
    Hex  = 0,
    Quad = 1,
    Line = 2,
    // Update count when adding new element types
    Count = 3
};

// Array containing all defined element types
inline constexpr auto element_types = std::invoke([] {
    auto retval = std::array< ElementTypes, static_cast< size_t >(ElementTypes::Count) >{};
    std::ranges::generate(retval, [i = 0]() mutable { return static_cast< ElementTypes >(i++); });
    return retval;
});
} // namespace lstr
#endif // L3STER_MESH_ELEMENTTYPES_HPP
