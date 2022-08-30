#ifndef L3STER_CUBEMESH_HPP
#define L3STER_CUBEMESH_HPP

#include "l3ster/mesh/Mesh.hpp"

namespace lstr
{
template < std::ranges::random_access_range Rx,
           std::ranges::random_access_range Ry,
           std::ranges::random_access_range Rz >
inline Mesh makeCubeMesh(Rx&& distx, Ry&& disty, Rz&& distz)
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

    MeshPartition::domain_map_t domains;
    domains[0].reserve< ElementTypes::Hex, 1 >(e_dx * e_dy * e_dz);
    domains[1].reserve< ElementTypes::Quad, 1 >(e_dx * e_dy);
    domains[2].reserve< ElementTypes::Quad, 1 >(e_dx * e_dy);
    domains[3].reserve< ElementTypes::Quad, 1 >(e_dx * e_dz);
    domains[4].reserve< ElementTypes::Quad, 1 >(e_dx * e_dz);
    domains[5].reserve< ElementTypes::Quad, 1 >(e_dz * e_dy);
    domains[6].reserve< ElementTypes::Quad, 1 >(e_dz * e_dy);

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
                domains[0].emplaceBack< ElementTypes::Hex, 1 >(nodes, verts, el_ind++);
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

            domains[1].emplaceBack< ElementTypes::Quad, 1 >(nodes1, verts1, el_ind++);
            domains[2].emplaceBack< ElementTypes::Quad, 1 >(nodes2, verts2, el_ind++);
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

            domains[3].emplaceBack< ElementTypes::Quad, 1 >(nodes1, verts1, el_ind++);
            domains[4].emplaceBack< ElementTypes::Quad, 1 >(nodes2, verts2, el_ind++);
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

            domains[5].emplaceBack< ElementTypes::Quad, 1 >(nodes1, verts1, el_ind++);
            domains[6].emplaceBack< ElementTypes::Quad, 1 >(nodes2, verts2, el_ind++);
        }
    }

    std::vector< n_id_t > nodes(n_dx * n_dy * n_dz);
    std::iota(begin(nodes), end(nodes), 0u);
    return Mesh{{MeshPartition{std::move(domains), std::move(nodes), std::vector< n_id_t >{}}}};
}

template < std::ranges::random_access_range R >
inline Mesh makeCubeMesh(R&& dist)
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeCubeMesh(dist, dist, dist);
}
} // namespace lstr
#endif // L3STER_CUBEMESH_HPP
