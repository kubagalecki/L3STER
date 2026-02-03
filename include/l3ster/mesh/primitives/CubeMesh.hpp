#ifndef L3STER_MESH_CUBEMESH_HPP
#define L3STER_MESH_CUBEMESH_HPP

#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"

namespace lstr::mesh
{
struct CubeMeshIds
{
    d_id_t domain = 0, back = 1, front = 2, bottom = 3, top = 4, left = 5, right = 6;
};

template < std::ranges::random_access_range Rx,
           std::ranges::random_access_range Ry,
           std::ranges::random_access_range Rz >
auto makeCubeMesh(Rx&& distx, Ry&& disty, Rz&& distz, const CubeMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >, val_t > and
             std::convertible_to< std::ranges::range_value_t< std::decay_t< Ry > >, val_t > and
             std::convertible_to< std::ranges::range_value_t< std::decay_t< Rz > >, val_t >
{
    const auto base_ids = SquareMeshIds{ids.domain, ids.bottom, ids.top, ids.left, ids.right};
    const auto base     = makeSquareMesh(std::forward< Rx >(distx), std::forward< Ry >(disty), base_ids);
    return extrude(base, std::forward< Rz >(distz), ids.back, ids.front);
}

template < std::ranges::random_access_range Rx,
           std::ranges::random_access_range Ry,
           std::ranges::random_access_range Rz >
auto makeCubeMeshQuadratic(Rx&& distx, Ry&& disty, Rz&& distz, const CubeMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >, val_t > and
             std::convertible_to< std::ranges::range_value_t< std::decay_t< Ry > >, val_t > and
             std::convertible_to< std::ranges::range_value_t< std::decay_t< Rz > >, val_t >
{
    const auto base_ids = SquareMeshIds{ids.domain, ids.bottom, ids.top, ids.left, ids.right};
    const auto base     = makeSquareMeshQuadratic(std::forward< Rx >(distx), std::forward< Ry >(disty), base_ids);
    return extrude(base, std::forward< Rz >(distz), ids.back, ids.front);
}

template < std::ranges::random_access_range R >
auto makeCubeMesh(R&& dist, const CubeMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeCubeMesh(dist, dist, dist, ids);
}

template < std::ranges::random_access_range R >
auto makeCubeMeshQuadratic(R&& dist, const CubeMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeCubeMeshQuadratic(dist, dist, dist, ids);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_CUBEMESH_HPP
