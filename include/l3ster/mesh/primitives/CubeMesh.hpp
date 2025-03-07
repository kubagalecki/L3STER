#ifndef L3STER_CUBEMESH_HPP
#define L3STER_CUBEMESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"

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
    const size_t n_dx = std::ranges::size(distx);
    const size_t n_dy = std::ranges::size(disty);
    const size_t n_dz = std::ranges::size(distz);
    const size_t e_dx = n_dx - 1;
    const size_t e_dy = n_dy - 1;
    const size_t e_dz = n_dz - 1;

    auto domains = MeshPartition< 1 >::domain_map_t{};
    domains[ids.domain].elements.getVector< Element< ElementType::Hex, 1 > >().reserve(e_dx * e_dy * e_dz);
    domains[ids.back].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dx * e_dy);
    domains[ids.front].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dx * e_dy);
    domains[ids.bottom].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dx * e_dz);
    domains[ids.top].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dx * e_dz);
    domains[ids.left].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dz * e_dy);
    domains[ids.right].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dz * e_dy);

    el_id_t el_ind = 0;

    // volume elements
    for (auto iz : std::views::iota(0u, e_dz))
    {
        for (auto iy : std::views::iota(0u, e_dy))
        {
            for (auto ix : std::views::iota(0u, e_dx))
            {
                const std::array< n_id_t, 8 >     nodes = {n_dx * n_dy * iz + n_dx * iy + ix,
                                                           n_dx * n_dy * iz + n_dx * iy + ix + 1,
                                                           n_dx * n_dy * iz + n_dx * (iy + 1) + ix,
                                                           n_dx * n_dy * iz + n_dx * (iy + 1) + ix + 1,
                                                           n_dx * n_dy * (iz + 1) + n_dx * iy + ix,
                                                           n_dx * n_dy * (iz + 1) + n_dx * iy + ix + 1,
                                                           n_dx * n_dy * (iz + 1) + n_dx * (iy + 1) + ix,
                                                           n_dx * n_dy * (iz + 1) + n_dx * (iy + 1) + ix + 1};
                const std::array< Point< 3 >, 8 > verts = {Point{distx[ix], disty[iy], distz[iz]},
                                                           Point{distx[ix + 1], disty[iy], distz[iz]},
                                                           Point{distx[ix], disty[iy + 1], distz[iz]},
                                                           Point{distx[ix + 1], disty[iy + 1], distz[iz]},
                                                           Point{distx[ix], disty[iy], distz[iz + 1]},
                                                           Point{distx[ix + 1], disty[iy], distz[iz + 1]},
                                                           Point{distx[ix], disty[iy + 1], distz[iz + 1]},
                                                           Point{distx[ix + 1], disty[iy + 1], distz[iz + 1]}};
                emplaceInDomain< ElementType::Hex, 1 >(domains[ids.domain], nodes, verts, el_ind++);
            }
        }
    }

    // z = const faces
    for (auto iy : std::views::iota(0u, e_dx))
    {
        for (auto ix : std::views::iota(0u, e_dx))
        {
            const std::array< n_id_t, 4 > nodes1 = {
                n_dx * iy + ix, n_dx * iy + ix + 1, n_dx * (iy + 1) + ix, n_dx * (iy + 1) + ix + 1};
            const std::array< Point< 3 >, 4 > verts1 = {Point{distx[ix], disty[iy], distz[0]},
                                                        Point{distx[ix + 1], disty[iy], distz[0]},
                                                        Point{distx[ix], disty[iy + 1], distz[0]},
                                                        Point{distx[ix + 1], disty[iy + 1], distz[0]}};

            auto nodes2 = nodes1;
            std::ranges::for_each(nodes2, [&](auto& n) { n += n_dx * n_dy * e_dz; });
            auto verts2 = verts1;
            std::ranges::for_each(verts2, [&](auto& v) { v = Point{v.x(), v.y(), distz[e_dz]}; });

            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.back], nodes1, verts1, el_ind++);
            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.front], nodes2, verts2, el_ind++);
        }
    }

    // y = const faces
    for (auto iz : std::views::iota(0u, e_dz))
    {
        for (auto ix : std::views::iota(0u, e_dx))
        {
            const std::array< n_id_t, 4 >     nodes1 = {n_dx * n_dy * iz + ix,
                                                        n_dx * n_dy * iz + ix + 1,
                                                        n_dx * n_dy * (iz + 1) + ix,
                                                        n_dx * n_dy * (iz + 1) + ix + 1};
            const std::array< Point< 3 >, 4 > verts1 = {Point{distx[ix], disty[0], distz[iz]},
                                                        Point{distx[ix + 1], disty[0], distz[iz]},
                                                        Point{distx[ix], disty[0], distz[iz + 1]},
                                                        Point{distx[ix + 1], disty[0], distz[iz + 1]}};

            auto nodes2 = nodes1;
            std::ranges::for_each(nodes2, [&](auto& n) { n += n_dx * e_dy; });
            auto verts2 = verts1;
            std::ranges::for_each(verts2, [&](auto& v) { v = Point{v.x(), disty[e_dy], v.z()}; });

            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.bottom], nodes1, verts1, el_ind++);
            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.top], nodes2, verts2, el_ind++);
        }
    }

    // x = const faces
    for (auto iz : std::views::iota(0u, e_dz))
    {
        for (auto iy : std::views::iota(0u, e_dy))
        {
            const std::array< n_id_t, 4 >     nodes1 = {n_dx * n_dy * iz + n_dx * iy,
                                                        n_dx * n_dy * iz + n_dx * (iy + 1),
                                                        n_dx * n_dy * (iz + 1) + n_dx * iy,
                                                        n_dx * n_dy * (iz + 1) + n_dx * (iy + 1)};
            const std::array< Point< 3 >, 4 > verts1 = {Point{distx[0], disty[iy], distz[iz]},
                                                        Point{distx[0], disty[iy + 1], distz[iz]},
                                                        Point{distx[0], disty[iy], distz[iz + 1]},
                                                        Point{distx[0], disty[iy + 1], distz[iz + 1]}};

            auto nodes2 = nodes1;
            std::ranges::for_each(nodes2, [&](auto& n) { n += e_dx; });
            auto verts2 = verts1;
            std::ranges::for_each(verts2, [&](auto& v) { v = Point{distx[e_dx], v.y(), v.z()}; });

            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.left], nodes1, verts1, el_ind++);
            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.right], nodes2, verts2, el_ind++);
        }
    }

    return {std::move(domains), 0, n_dx * n_dy * n_dz, {ids.back, ids.front, ids.bottom, ids.top, ids.left, ids.right}};
}

template < std::ranges::random_access_range R >
auto makeCubeMesh(R&& dist, const CubeMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeCubeMesh(dist, dist, dist, ids);
}
} // namespace lstr::mesh
#endif // L3STER_CUBEMESH_HPP
