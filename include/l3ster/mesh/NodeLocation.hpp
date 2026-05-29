#ifndef L3STER_MESH_NODELOCATION_HPP
#define L3STER_MESH_NODELOCATION_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/common/Structs.hpp"

namespace lstr::mesh
{
namespace detail
{
template < el_o_t O >
auto makeLineNodeLocations()
{
    return util::elwise(math::getLobattoRuleAbsc< val_t, O + 1 >(), [](auto x) { return Point{x}; });
}

template < el_o_t O >
auto makeQuadNodeLocations()
{
    constexpr auto nodes_per_edge = O + 1;
    const auto&    absc           = math::getLobattoRuleAbsc< val_t, nodes_per_edge >();
    const auto     points         = std::views::cartesian_product(absc, absc);
    auto           retval         = std::array< Point< 2 >, nodes_per_edge * nodes_per_edge >{};
    std::ranges::transform(points, retval.begin(), [](auto xy) {
        auto [y, x] = xy;
        return Point(x, y);
    });
    return retval;
}

template < el_o_t O >
auto makeHexNodeLocations()
{
    constexpr auto nodes_per_edge = O + 1;
    const auto&    absc           = math::getLobattoRuleAbsc< val_t, nodes_per_edge >();
    const auto     points         = std::views::cartesian_product(absc, absc, absc);
    auto           retval         = std::array< Point< 3 >, nodes_per_edge * nodes_per_edge * nodes_per_edge >{};
    std::ranges::transform(points, retval.begin(), [](auto xyz) {
        auto [z, y, x] = xyz;
        return Point(x, y, z);
    });
    return retval;
}
} // namespace detail

template < ElementType T, el_o_t O >
const auto& getNodeLocations()
{
    constexpr auto GT = ElementTraits< Element< T, O > >::geom_type;
    if constexpr (GT == ElementType::Line)
    {
        static const auto value = detail::makeLineNodeLocations< O >();
        return value;
    }
    else if constexpr (GT == ElementType::Quad)
    {
        static const auto value = detail::makeQuadNodeLocations< O >();
        return value;
    }
    else if constexpr (GT == ElementType::Hex)
    {
        static const auto value = detail::makeHexNodeLocations< O >();
        return value;
    }
    else
        static_assert(util::always_false< T >);
}

template < ElementType T, el_o_t O >
auto getSideNodeLocations(el_side_t side)
{
    using eltraits           = ElementTraits< Element< T, O > >;
    const auto all_node_locs = getNodeLocations< T, O >();
    return util::elwise(eltraits::boundary_table[side], [&](auto i) { return all_node_locs[i]; });
}
} // namespace lstr::mesh
#endif // L3STER_MESH_NODELOCATION_HPP
