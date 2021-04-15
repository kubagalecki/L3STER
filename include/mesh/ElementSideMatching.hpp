#ifndef L3STER_MESH_ELEMENTSIDEMATCHING_HPP
#define L3STER_MESH_ELEMENTSIDEMATCHING_HPP
#include "mesh/ElementTraits.hpp"

namespace lstr::detail
{
template < el_ns_t I, typename Element, size_t N >
constexpr auto matchSide(const Element& element, const std::array< n_id_t, N >& sorted_side_nodes)
{
    constexpr auto& side_inds = std::get< I >(ElementTraits< Element >::boundary_table);
    if constexpr (std::tuple_size_v< std::decay_t< decltype(side_inds) > > == N)
    {
        std::array< n_id_t, N > element_side_nodes;
        std::ranges::transform(
            side_inds, element_side_nodes.begin(), [&](el_locind_t i) { return element.getNodes()[i]; });
        std::ranges::sort(element_side_nodes);
        return std::ranges::equal(element_side_nodes, sorted_side_nodes);
    }
    else
        return false;
}

template < el_ns_t I, typename Element, size_t N >
constexpr el_ns_t matchSidesRecursively(const Element& element, const std::array< n_id_t, N >& sorted_side_nodes)
{
    if (matchSide< I >(element, sorted_side_nodes))
        return I;
    else if constexpr (I > 0)
        return matchSidesRecursively< I - 1 >(element, sorted_side_nodes);
    else if (matchSide< 0 >(element, sorted_side_nodes))
        return static_cast< el_ns_t >(0u);
    else
        return std::numeric_limits< el_ns_t >::max();
}

template < ElementTypes T, el_o_t O, size_t N >
constexpr el_ns_t matchBoundaryNodesToElement(const Element< T, O >&         element,
                                              const std::array< n_id_t, N >& sorted_boundary_nodes)
{
    constexpr auto n_sides = ElementTraits< Element< T, O > >::n_sides;
    return matchSidesRecursively< n_sides - 1 >(element, sorted_boundary_nodes);
}
} // namespace lstr::detail
#endif // L3STER_MESH_ELEMENTSIDEMATCHING_HPP
