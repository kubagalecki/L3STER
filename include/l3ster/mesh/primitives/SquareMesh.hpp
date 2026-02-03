#ifndef L3STER_MESH_SQUAREMESH_HPP
#define L3STER_MESH_SQUAREMESH_HPP

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
    constexpr auto line = ElementType::Line;
    constexpr auto quad = ElementType::Quad;

    const size_t n_dx = std::ranges::size(distx);
    const size_t n_dy = std::ranges::size(disty);
    const size_t e_dx = n_dx - 1;
    const size_t e_dy = n_dy - 1;

    auto domains = MeshPartition< 1 >::domain_map_t{};
    domains[ids.domain].elements.getVector< Element< quad, 1 > >().reserve(e_dx * e_dy);
    domains[ids.bottom].elements.getVector< Element< line, 1 > >().reserve(e_dx);
    domains[ids.top].elements.getVector< Element< line, 1 > >().reserve(e_dx);
    domains[ids.left].elements.getVector< Element< line, 1 > >().reserve(e_dy);
    domains[ids.right].elements.getVector< Element< line, 1 > >().reserve(e_dy);

    el_id_t el_ind = 0;

    // Area elements
    for (auto [iy, ix] : std::views::cartesian_product(std::views::iota(0u, e_dy), std::views::iota(0u, e_dx)))
    {
        const auto nodes =
            std::array{n_dx * iy + ix, n_dx * iy + ix + 1, n_dx * (iy + 1) + ix, n_dx * (iy + 1) + ix + 1};
        const auto x1 = distx[ix], x2 = distx[ix + 1], y1 = disty[iy], y2 = disty[iy + 1];
        const auto verts = std::array{Point{x1, y1, 0.}, Point{x2, y1, 0.}, Point{x1, y2, 0.}, Point{x2, y2, 0.}};
        emplaceInDomain< quad, 1 >(domains[ids.domain], nodes, verts, el_ind++);
    }

    // y = const faces
    for (auto ix : std::views::iota(0u, e_dx))
    {
        const auto nodes1 = std::array< n_id_t, 2 >{ix, ix + 1};
        const auto nodes2 = util::elwise(nodes1, std::bind_back(std::plus{}, n_dx * e_dy));
        const auto verts1 = std::array{Point{distx[ix], disty[0], 0.}, Point{distx[ix + 1], disty[0], 0.}};
        const auto verts2 = std::array{Point{distx[ix], disty[e_dy], 0.}, Point{distx[ix + 1], disty[e_dy], 0.}};

        emplaceInDomain< line, 1 >(domains[ids.bottom], nodes1, verts1, el_ind++);
        emplaceInDomain< line, 1 >(domains[ids.top], nodes2, verts2, el_ind++);
    }

    // x = const faces
    for (auto iy : std::views::iota(0u, e_dy))
    {
        const auto nodes1 = std::array< n_id_t, 2 >{iy * n_dx, (iy + 1) * n_dx};
        const auto nodes2 = util::elwise(nodes1, std::bind_back(std::plus{}, e_dx));
        const auto verts1 = std::array{Point{distx[0], disty[iy], 0.}, Point{distx[0], disty[iy + 1], 0.}};
        const auto verts2 = std::array{Point{distx[e_dx], disty[iy], 0.}, Point{distx[e_dx], disty[iy + 1], 0.}};

        emplaceInDomain< line, 1 >(domains[ids.left], nodes1, verts1, el_ind++);
        emplaceInDomain< line, 1 >(domains[ids.right], nodes2, verts2, el_ind++);
    }

    return {std::move(domains), 0, n_dx * n_dy, {ids.bottom, ids.top, ids.left, ids.right}};
}

template < std::ranges::random_access_range R >
auto makeSquareMesh(R&& dist, const SquareMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeSquareMesh(dist, dist, ids);
}

template < std::ranges::random_access_range Rx, std::ranges::random_access_range Ry >
auto makeSquareMeshQuadratic(Rx&& distx, Ry&& disty, const SquareMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >, val_t > and
             std::convertible_to< std::ranges::range_value_t< std::decay_t< Ry > >, val_t >
{
    constexpr auto line = ElementType::Line2;
    constexpr auto quad = ElementType::Quad2;

    const size_t n_dx = std::ranges::size(distx);
    const size_t n_dy = std::ranges::size(disty);
    const size_t e_dx = n_dx - 1;
    const size_t e_dy = n_dy - 1;

    auto domains = MeshPartition< 1 >::domain_map_t{};
    domains[ids.domain].elements.getVector< Element< quad, 1 > >().reserve(e_dx * e_dy);
    domains[ids.bottom].elements.getVector< Element< line, 1 > >().reserve(e_dx);
    domains[ids.top].elements.getVector< Element< line, 1 > >().reserve(e_dx);
    domains[ids.left].elements.getVector< Element< line, 1 > >().reserve(e_dy);
    domains[ids.right].elements.getVector< Element< line, 1 > >().reserve(e_dy);

    el_id_t el_ind = 0;

    // Area elements
    for (auto [iy, ix] : std::views::cartesian_product(std::views::iota(0u, e_dy), std::views::iota(0u, e_dx)))
    {
        const auto nodes =
            std::array{n_dx * iy + ix, n_dx * iy + ix + 1, n_dx * (iy + 1) + ix, n_dx * (iy + 1) + ix + 1};
        const auto x1 = distx[ix], x3 = distx[ix + 1], x2 = std::midpoint(x1, x3);
        const auto y1 = disty[iy], y3 = disty[iy + 1], y2 = std::midpoint(y1, y3);
        const auto verts = std::array{Point{x1, y1, 0.},
                                      Point{x2, y1, 0.},
                                      Point{x3, y1, 0.},
                                      Point{x1, y2, 0.},
                                      Point{x2, y2, 0.},
                                      Point{x3, y2, 0.},
                                      Point{x1, y3, 0.},
                                      Point{x2, y3, 0.},
                                      Point{x3, y3, 0.}};
        emplaceInDomain< quad, 1 >(domains[ids.domain], nodes, verts, el_ind++);
    }

    // y = const faces
    for (auto ix : std::views::iota(0u, e_dx))
    {
        const auto ymin = disty[0], ymax = disty[e_dy], x1 = distx[ix], x3 = distx[ix + 1], x2 = std::midpoint(x1, x3);
        const auto nodes1 = std::array< n_id_t, 2 >{ix, ix + 1};
        const auto nodes2 = util::elwise(nodes1, std::bind_back(std::plus{}, n_dx * e_dy));
        const auto verts1 = std::array{Point{x1, ymin, 0.}, Point{x2, ymin, 0.}, Point{x3, ymin, 0.}};
        const auto verts2 = std::array{Point{x1, ymax, 0.}, Point{x2, ymax, 0.}, Point{x3, ymax, 0.}};

        emplaceInDomain< line, 1 >(domains[ids.bottom], nodes1, verts1, el_ind++);
        emplaceInDomain< line, 1 >(domains[ids.top], nodes2, verts2, el_ind++);
    }

    // x = const faces
    for (auto iy : std::views::iota(0u, e_dy))
    {
        const auto xmin = distx[0], xmax = distx[e_dx], y1 = disty[iy], y3 = disty[iy + 1], y2 = std::midpoint(y1, y3);
        const auto nodes1 = std::array< n_id_t, 2 >{iy * n_dx, (iy + 1) * n_dx};
        const auto nodes2 = util::elwise(nodes1, std::bind_back(std::plus{}, e_dx));
        const auto verts1 = std::array{Point{xmin, y1, 0.}, Point{xmin, y2, 0.}, Point{xmin, y3, 0.}};
        const auto verts2 = std::array{Point{xmax, y1, 0.}, Point{xmax, y2, 0.}, Point{xmax, y3, 0.}};

        emplaceInDomain< line, 1 >(domains[ids.left], nodes1, verts1, el_ind++);
        emplaceInDomain< line, 1 >(domains[ids.right], nodes2, verts2, el_ind++);
    }

    return {std::move(domains), 0, n_dx * n_dy, {ids.bottom, ids.top, ids.left, ids.right}};
}

template < std::ranges::random_access_range R >
auto makeSquareMeshQuadratic(R&& dist, const SquareMeshIds& ids = {}) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< R > >, val_t >
{
    return makeSquareMeshQuadratic(dist, dist, ids);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_SQUAREMESH_HPP
