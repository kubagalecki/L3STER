#ifndef L3STER_MESH_CONSTANTS_HPP
#define L3STER_MESH_CONSTANTS_HPP

#include "defs/Typedefs.h"

#include <array>

namespace lstr
{
// Define allowed spatial dimensions
constexpr inline std::array< dim_t, 3 > space_dims = {1, 2, 3};

// Define allowed element orders
constexpr inline std::array< el_o_t, 2 > element_orders = {1, 2};
} // namespace lstr

#endif // L3STER_MESH_CONSTANTS_HPP