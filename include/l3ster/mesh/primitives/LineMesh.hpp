#ifndef L3STER_LINEMESH_HPP
#define L3STER_LINEMESH_HPP

#include "l3ster/mesh/MeshPartition.hpp"

namespace lstr::mesh
{
template < std::ranges::random_access_range Rx >
auto makeLineMesh(Rx&& distx) -> MeshPartition< 1 >
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >, val_t >
{
    const size_t n_dx = std::ranges::size(distx);
    const size_t e_dx = n_dx - 1;

    auto domains = MeshPartition< 1 >::domain_map_t{};
    domains[0].elements.getVector< Element< ElementType::Line, 1 > >().reserve(e_dx);

    for (el_id_t id : std::views::iota(0u, e_dx))
    {
        const std::array< n_id_t, 2 >     nodes = {id, id + 1};
        const std::array< Point< 3 >, 2 > verts = {Point{distx[id], 0., 0.}, Point{distx[id + 1], 0., 0.}};
        emplaceInDomain< ElementType::Line, 1 >(domains[0], nodes, verts, id);
    }

    return {std::move(domains), 0, n_dx, {}};
}
} // namespace lstr::mesh
#endif // L3STER_LINEMESH_HPP
