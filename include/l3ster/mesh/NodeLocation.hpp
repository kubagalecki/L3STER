#ifndef L3STER_MESH_NODELOCATION_HPP
#define L3STER_MESH_NODELOCATION_HPP

#include "l3ster/common/Structs.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/math/LobattoRuleAbsc.hpp"

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

namespace detail
{
template < ElementType T, el_o_t O >
const auto& getGeomBasisValuesAtNodes()
{
    static const auto value = std::invoke([] {
        using elem_traits            = ElementTraits< Element< T, O > >;
        constexpr auto num_nodes     = elem_traits::nodes_per_element;
        constexpr auto GBT           = basis::BasisType::Lagrange;
        constexpr auto GT            = elem_traits::geom_type;
        constexpr auto GO            = elem_traits::geom_order;
        constexpr auto num_bases     = Element< GT, GO >::n_nodes;
        const auto&    node_ref_locs = getNodeLocations< GT, O >();
        auto           retval        = Eigen::Matrix< val_t, num_bases, num_nodes >{};
        for (auto&& [i, x] : node_ref_locs | std::views::enumerate)
            retval.col(i) = basis::computeReferenceBases< GT, GO, GBT >(x);
        return retval;
    });
    return value;
}
} // namespace detail

template < ElementType T, el_o_t O >
Point< 3 > nodePhysicalLocation(const Element< T, O >& element, el_locind_t i)
{
    const auto  verts      = element.data.getEigenMap();
    const auto& geom_basis = detail::getGeomBasisValuesAtNodes< T, O >();
    const auto  phys_loc   = (verts * geom_basis.col(i)).eval();
    return Point{phys_loc[0], phys_loc[1], phys_loc[2]};
}

template < ElementType T, el_o_t O >
auto nodePhysicalLocation(const Element< T, O >& element)
{
    constexpr auto n_nodes                     = Element< T, O >::n_nodes;
    using retval_map_t                         = Eigen::Map< Eigen::Matrix< val_t, 3, n_nodes > >;
    const auto  verts                          = element.data.getEigenMap();
    const auto& geom_basis                     = detail::getGeomBasisValuesAtNodes< T, O >();
    auto        retval                         = std::array< Point< 3 >, n_nodes >{};
    retval_map_t{retval.front().coords.data()} = verts * geom_basis;
    return retval;
}
} // namespace lstr::mesh
#endif // L3STER_MESH_NODELOCATION_HPP
