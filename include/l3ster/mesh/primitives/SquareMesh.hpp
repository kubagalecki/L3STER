#ifndef L3STER_SQUAREMESH_HPP
#define L3STER_SQUAREMESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"

namespace lstr::mesh
{
struct SquareMeshIds
{
    d_id_t domain = 0, bottom = 1, top = 2, left = 3, right = 4;
};

template < std::ranges::random_access_range Rx, std::ranges::random_access_range Ry >
auto makeSquareMesh(Rx&& distx, Ry&& disty, const SquareMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >, val_t > and
             std::convertible_to< std::ranges::range_value_t< std::decay_t< Ry > >, val_t >
{
    const size_t n_dx = std::ranges::size(distx);
    const size_t n_dy = std::ranges::size(disty);
    const size_t e_dx = n_dx - 1;
    const size_t e_dy = n_dy - 1;

    auto domains = MeshPartition< 1 >::domain_map_t{};
    domains[ids.domain].elements.getVector< Element< ElementType::Quad, 1 > >().reserve(e_dx * e_dy);
    domains[ids.bottom].elements.getVector< Element< ElementType::Line, 1 > >().reserve(e_dx);
    domains[ids.top].elements.getVector< Element< ElementType::Line, 1 > >().reserve(e_dx);
    domains[ids.left].elements.getVector< Element< ElementType::Line, 1 > >().reserve(e_dy);
    domains[ids.right].elements.getVector< Element< ElementType::Line, 1 > >().reserve(e_dy);

    el_id_t el_ind = 0;

    // Area elements
    for (auto iy : std::views::iota(0u, e_dy))
        for (auto ix : std::views::iota(0u, e_dx))
        {
            const std::array< n_id_t, 4 > nodes = {
                n_dx * iy + ix, n_dx * iy + ix + 1, n_dx * (iy + 1) + ix, n_dx * (iy + 1) + ix + 1};
            const std::array< Point< 3 >, 4 > verts = {Point{distx[ix], disty[iy], 0.},
                                                       Point{distx[ix + 1], disty[iy], 0.},
                                                       Point{distx[ix], disty[iy + 1], 0.},
                                                       Point{distx[ix + 1], disty[iy + 1], 0.}};
            emplaceInDomain< ElementType::Quad, 1 >(domains[ids.domain], nodes, verts, el_ind++);
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

        emplaceInDomain< ElementType::Line, 1 >(domains[ids.bottom], nodes1, verts1, el_ind++);
        emplaceInDomain< ElementType::Line, 1 >(domains[ids.top], nodes2, verts2, el_ind++);
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

        emplaceInDomain< ElementType::Line, 1 >(domains[ids.left], nodes1, verts1, el_ind++);
        emplaceInDomain< ElementType::Line, 1 >(domains[ids.right], nodes2, verts2, el_ind++);
    }

    return {std::move(domains), 0, n_dx * n_dy, {ids.bottom, ids.top, ids.left, ids.right}};
}

template < std::ranges::random_access_range R >
auto makeSquareMesh(R&& dist, const SquareMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeSquareMesh(dist, dist, ids);
}
} // namespace lstr::mesh
#endif // L3STER_SQUAREMESH_HPP
