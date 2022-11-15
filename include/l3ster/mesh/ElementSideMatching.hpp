#ifndef L3STER_MESH_ELEMENTSIDEMATCHING_HPP
#define L3STER_MESH_ELEMENTSIDEMATCHING_HPP

#include "l3ster/mesh/ElementTraits.hpp"
#include "l3ster/util/Algorithm.hpp"

namespace lstr::detail
{
template < el_side_t I, typename Element, size_t N >
constexpr bool matchSide(const Element& element, const std::array< n_id_t, N >& sorted_side_nodes)
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

template < el_side_t I, typename Element, size_t N >
constexpr el_side_t matchSidesRecursively(const Element& element, const std::array< n_id_t, N >& sorted_side_nodes)
{
    if (matchSide< I >(element, sorted_side_nodes))
        return I;
    else if constexpr (I > 0)
        return matchSidesRecursively< I - 1 >(element, sorted_side_nodes);
    else if (matchSide< 0 >(element, sorted_side_nodes))
        return 0;
    else
        return std::numeric_limits< el_side_t >::max();
}

template < ElementTypes T, el_o_t O, size_t N >
constexpr el_side_t matchBoundaryNodesToElement(const Element< T, O >&         element,
                                                const std::array< n_id_t, N >& srt_boundary_nodes)
{
    const auto              srt_element_nodes = getSortedArray(element.getNodes());
    std::array< n_id_t, N > common;
    const auto common_end = std::ranges::set_intersection(srt_element_nodes, srt_boundary_nodes, begin(common)).out;
    if (common_end == begin(common)) [[likely]]
        return std::numeric_limits< el_side_t >::max();

    constexpr auto n_sides = ElementTraits< Element< T, O > >::n_sides;
    return matchSidesRecursively< n_sides - 1 >(element, srt_boundary_nodes);
}
} // namespace lstr::detail
#endif // L3STER_MESH_ELEMENTSIDEMATCHING_HPP
