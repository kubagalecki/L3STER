#ifndef L3STER_SQUAREMESH_HPP
#define L3STER_SQUAREMESH_HPP

#include "l3ster/mesh/Mesh.hpp"

namespace lstr
{
template < std::ranges::random_access_range Rx, std::ranges::random_access_range Ry >
inline Mesh makeSquareMesh(Rx&& distx, Ry&& disty)
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >,
                                  val_t >and std::convertible_to< std::ranges::range_value_t< std::decay_t< Ry > >,
                                                                  val_t >
{
    const size_t n_dx = std::ranges::size(distx);
    const size_t n_dy = std::ranges::size(disty);
    const size_t e_dx = n_dx - 1;
    const size_t e_dy = n_dy - 1;

    MeshPartition::domain_map_t domains;
    domains[0].reserve< ElementTypes::Quad, 1 >(e_dx * e_dy);
    domains[1].reserve< ElementTypes::Line, 1 >(e_dx);
    domains[2].reserve< ElementTypes::Line, 1 >(e_dx);
    domains[3].reserve< ElementTypes::Line, 1 >(e_dy);
    domains[4].reserve< ElementTypes::Line, 1 >(e_dy);

    el_id_t el_ind = 0;

    // area elements
    for (auto iy : std::views::iota(0u, e_dy))
    {
        for (auto ix : std::views::iota(0u, e_dx))
        {
            const std::array< n_id_t, 4 > nodes = {
                n_dx * iy + ix, n_dx * iy + ix + 1, n_dx * (iy + 1) + ix, n_dx * (iy + 1) + ix + 1};
            const std::array< Point< 3 >, 4 > verts = {Point{distx[ix], disty[iy], 0.},
                                                       Point{distx[ix + 1], disty[iy], 0.},
                                                       Point{distx[ix], disty[iy + 1], 0.},
                                                       Point{distx[ix + 1], disty[iy + 1], 0.}};
            domains[0].emplaceBack< ElementTypes::Quad, 1 >(nodes, verts, el_ind++);
        }
    }

    // y = const faces
    for (auto ix : std::views::iota(0u, e_dx))
    {
        const std::array< n_id_t, 2 >     nodes1 = {ix, ix + 1};
        const std::array< Point< 3 >, 2 > verts1 = {Point{distx[ix], disty[0], 0.}, Point{distx[ix + 1], disty[0], 0.}};

        auto nodes2 = nodes1;
        std::ranges::for_each(nodes2, [&](auto& n) { n += n_dx * e_dy; });
        auto verts2 = verts1;
        std::ranges::for_each(verts2, [&](auto& v) { v = Point{v.x(), disty[e_dy], 0.}; });

        domains[1].emplaceBack< ElementTypes::Line, 1 >(nodes1, verts1, el_ind++);
        domains[2].emplaceBack< ElementTypes::Line, 1 >(nodes2, verts2, el_ind++);
    }

    // x = const faces
    for (auto iy : std::views::iota(0u, e_dy))
    {
        const std::array< n_id_t, 2 >     nodes1 = {iy * n_dx, (iy + 1) * n_dx};
        const std::array< Point< 3 >, 2 > verts1 = {Point{distx[0], disty[iy], 0.}, Point{distx[0], disty[iy + 1], 0.}};

        auto nodes2 = nodes1;
        std::ranges::for_each(nodes2, [&](auto& n) { n += e_dx; });
        auto verts2 = verts1;
        std::ranges::for_each(verts2, [&](auto& v) { v = Point{distx[e_dx], v.y(), 0.}; });

        domains[3].emplaceBack< ElementTypes::Line, 1 >(nodes1, verts1, el_ind++);
        domains[4].emplaceBack< ElementTypes::Line, 1 >(nodes2, verts2, el_ind++);
    }

    std::vector< n_id_t > nodes(n_dx * n_dy);
    std::iota(begin(nodes), end(nodes), 0u);
    return Mesh{{MeshPartition{std::move(domains), std::move(nodes), std::vector< n_id_t >{}}}};
}

template < std::ranges::random_access_range R >
inline Mesh makeSquareMesh(R&& dist)
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >,
                                  val_t >
{
    return makeSquareMesh(dist, dist);
}
} // namespace lstr
#endif // L3STER_SQUAREMESH_HPP