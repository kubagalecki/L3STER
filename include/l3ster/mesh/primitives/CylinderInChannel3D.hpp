#ifndef L3STER_MESH_CYLINDERINCHANNEL3D_HPP
#define L3STER_MESH_CYLINDERINCHANNEL3D_HPP

#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/mesh/primitives/CylinderInChannel2D.hpp"

namespace lstr::mesh
{
template < std::ranges::random_access_range R >
auto makeCylinderInChannel3DMesh(R&&                                zdist,
                                 const CylinderInChannel2DGeometry& geometry = {},
                                 const CylinderInChannel2DMeshIds&  ids      = {},
                                 d_id_t                             id_back  = 6,
                                 d_id_t                             id_front = 7) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    const auto mesh2d = makeCylinderInChannel2DMesh(geometry, ids);
    return extrude(mesh2d, std::forward< R >(zdist), id_back, id_front);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_CYLINDERINCHANNEL3D_HPP
