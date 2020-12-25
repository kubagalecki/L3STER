#ifndef L3STER_MESH_CONSTANTS_HPP
#define L3STER_MESH_CONSTANTS_HPP

#include "defs/Typedefs.h"

#include <array>

namespace lstr::mesh
{
// Define allowed spatial dimensions
constexpr inline std::array< types::dim_t, 3 > space_dims = {1, 2, 3};

// Define allowed element orders
constexpr inline std::array< types::el_o_t, 2 > element_orders = {1, 2};
} // namespace lstr::mesh

#endif // L3STER_MESH_CONSTANTS_HPP