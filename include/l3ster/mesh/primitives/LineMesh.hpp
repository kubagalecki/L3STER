#ifndef L3STER_LINEMESH_HPP
#define L3STER_LINEMESH_HPP

#include "l3ster/mesh/Mesh.hpp"

namespace lstr
{
template < std::ranges::random_access_range Rx >
inline Mesh makeLineMesh(Rx&& distx)
    requires std::convertible_to< std::ranges::range_value_t< std::decay_t< Rx > >,
                                  val_t >
{
    const size_t n_dx = std::ranges::size(distx);
    const size_t e_dx = n_dx - 1;

    MeshPartition::domain_map_t domains;
    domains[0].reserve< ElementTypes::Line, 1 >(e_dx);

    for (el_id_t id : std::views::iota(0u, e_dx))
    {
        const std::array< n_id_t, 2 >     nodes = {id, id + 1};
        const std::array< Point< 3 >, 2 > verts = {Point{distx[id], 0., 0.}, Point{distx[id + 1], 0., 0.}};
        domains[0].emplaceBack< ElementTypes::Line, 1 >(nodes, verts, id);
    }

    std::vector< n_id_t > nodes(n_dx);
    std::iota(begin(nodes), end(nodes), 0);
    return Mesh{{MeshPartition{std::move(domains), std::move(nodes), std::vector< n_id_t >{}}}};
}
} // namespace lstr
#endif // L3STER_LINEMESH_HPP
