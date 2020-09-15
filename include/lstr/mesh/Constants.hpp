#ifndef L3STER_MESH_CONSTANTS_HPP
#define L3STER_MESH_CONSTANTS_HPP

#include "lstr/defs/Typedefs.h"
#include "lstr/util/Meta.hpp"

#include <array>

namespace lstr::mesh
{
// Define allowed spatial dimensions
constexpr inline std::array< types::dim_t, 3 > allowed_dims = {1, 2, 3};

// Define allowed element orders
constexpr inline std::array< types::el_o_t, 2 > allowed_orders = {1, 2};
} // namespace lstr::mesh

namespace lstr::mesh
{
struct MeshDimensionsArray
{
    static constexpr auto values = allowed_orders;
};

template < template < typename... > typename T, template < auto > typename U >
using parametrize_over_dims_t = typename util::meta::
    apply_valseq< T, U, typename util::meta::array_to_valseq< MeshDimensionsArray >::type >::type;
} // namespace lstr::mesh

#endif // L3STER_MESH_CONSTANTS_HPP