#ifndef L3STER_MESH_ELEMENTTRAITS_HPP
#define L3STER_MESH_ELEMENTTRAITS_HPP

#include "l3ster/common/Structs.hpp"
#include "l3ster/mesh/ElementType.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Concepts.hpp"

#include <span>

namespace lstr::mesh
{
namespace detail
{
template < typename BoundaryTable >
consteval auto getBoundaryInds(const BoundaryTable& boundary_table) -> std::vector< el_locind_t >
{
    auto retval = boundary_table | std::views::join | std::ranges::to< std::vector >();
    util::sortRemoveDup(retval);
    return retval;
}
template < auto boundary_table >
consteval auto getBoundaryNodeInds()
{
    constexpr auto num_boundary_inds = getBoundaryInds(boundary_table).size();
    auto           retval            = std::array< el_locind_t, num_boundary_inds >{};
    std::ranges::copy(getBoundaryInds(boundary_table), retval.begin());
    return retval;
}
template < size_t num_nodes, auto boundary_inds >
consteval auto getInternalNodeInds()
{
    constexpr auto num_internal_inds = num_nodes - boundary_inds.size();
    auto           retval            = std::array< el_locind_t, num_internal_inds >{};
    constexpr auto all_inds          = std::views::iota(el_locind_t{}, num_nodes);
    std::ranges::set_difference(all_inds, boundary_inds, retval.begin());
    return retval;
}

template < el_o_t O >
struct HexTraitsBase
{
    static constexpr ElementType geom_type          = ElementType::Hex;
    static constexpr el_o_t      order              = O;
    static constexpr el_locind_t nodes_per_element  = (O + 1) * (O + 1) * (O + 1);
    static constexpr dim_t       native_dim         = 3;
    static constexpr el_side_t   n_sides            = 6;
    static constexpr auto        boundary_table     = std::invoke([] {
        using lid                       = el_locind_t;
        constexpr size_t nodes_per_edge = O + 1;
        constexpr auto   nodes_per_side = nodes_per_edge * nodes_per_edge;
        constexpr auto   back_shift     = nodes_per_side * (nodes_per_edge - 1);
        constexpr auto   top_shift      = nodes_per_edge * (nodes_per_edge - 1);
        constexpr auto   right_shift    = nodes_per_edge - 1;
        constexpr auto   iterspace1d    = std::views::iota(0uz, nodes_per_edge);
        auto             retval         = std::array< std::array< el_locind_t, (O + 1) * (O + 1) >, n_sides >{};
        for (auto&& [index, ij] : std::views::cartesian_product(iterspace1d, iterspace1d) | std::views::enumerate)
        {
            const auto induz  = static_cast< size_t >(index);
            const auto [i, j] = ij;
            retval[0][induz]  = static_cast< lid >(induz);                                   // front
            retval[1][induz]  = static_cast< lid >(retval[0][induz] + back_shift);           // back
            retval[2][induz]  = static_cast< lid >(i * nodes_per_side + j);                  // bottom
            retval[3][induz]  = static_cast< lid >(retval[2][induz] + top_shift);            // top
            retval[4][induz]  = static_cast< lid >(i * nodes_per_side + j * nodes_per_edge); // left
            retval[5][induz]  = static_cast< lid >(retval[4][induz] + right_shift);          // right
        }
        return retval;
    });
    static constexpr auto        boundary_node_inds = getBoundaryNodeInds< boundary_table >();
    static constexpr auto        internal_node_inds = getInternalNodeInds< nodes_per_element, boundary_node_inds >();
};

template < el_o_t O >
struct QuadTraitsBase
{
    static constexpr ElementType geom_type          = ElementType::Quad;
    static constexpr el_o_t      order              = O;
    static constexpr el_locind_t nodes_per_element  = (O + 1) * (O + 1);
    static constexpr dim_t       native_dim         = 2;
    static constexpr el_side_t   n_sides            = 4;
    static constexpr auto        boundary_table     = std::invoke([] {
        auto             retval         = std::array< std::array< el_locind_t, O + 1 >, n_sides >{};
        constexpr size_t nodes_per_side = O + 1;
        constexpr auto   top_shift      = nodes_per_side * (nodes_per_side - 1);
        constexpr auto   right_shift    = nodes_per_side - 1;
        for (size_t i = 0; i < nodes_per_side; ++i)
        {
            retval[0][i] = static_cast< el_locind_t >(i);                          // bottom
            retval[1][i] = static_cast< el_locind_t >(retval[0][i] + top_shift);   // top
            retval[2][i] = static_cast< el_locind_t >(i * nodes_per_side);         // left
            retval[3][i] = static_cast< el_locind_t >(retval[2][i] + right_shift); // right
        }
        return retval;
    });
    static constexpr auto        boundary_node_inds = getBoundaryNodeInds< boundary_table >();
    static constexpr auto        internal_node_inds = getInternalNodeInds< nodes_per_element, boundary_node_inds >();
};

template < el_o_t O >
struct LineTraitsBase
{
    static constexpr ElementType geom_type          = ElementType::Line;
    static constexpr el_o_t      order              = O;
    static constexpr el_locind_t nodes_per_element  = O + 1;
    static constexpr dim_t       native_dim         = 1;
    static constexpr el_side_t   n_sides            = 2;
    static constexpr auto        boundary_table     = std::array{std::array{el_locind_t{}}, std::array{el_locind_t{O}}};
    static constexpr auto        boundary_node_inds = getBoundaryNodeInds< boundary_table >();
    static constexpr auto        internal_node_inds = getInternalNodeInds< nodes_per_element, boundary_node_inds >();
};
} // namespace detail

template < ElementType ET, el_o_t EO >
struct Element;

template < typename >
struct ElementTraits;

template < el_o_t O >
struct ElementTraits< Element< ElementType::Hex, O > > : detail::HexTraitsBase< O >
{
    static constexpr auto   type       = ElementType::Hex;
    static constexpr el_o_t geom_order = 1;
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Hex2, O > > : detail::HexTraitsBase< O >
{
    static constexpr auto   type       = ElementType::Hex2;
    static constexpr el_o_t geom_order = 2;
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Quad, O > > : detail::QuadTraitsBase< O >
{
    static constexpr auto   type       = ElementType::Quad;
    static constexpr el_o_t geom_order = 1;
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Quad2, O > > : detail::QuadTraitsBase< O >
{
    static constexpr auto   type       = ElementType::Quad2;
    static constexpr el_o_t geom_order = 2;
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Line, O > > : detail::LineTraitsBase< O >
{
    static constexpr auto   type       = ElementType::Line;
    static constexpr el_o_t geom_order = 1;
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Line2, O > > : detail::LineTraitsBase< O >
{
    static constexpr auto   type       = ElementType::Line2;
    static constexpr el_o_t geom_order = 2;
};

template < ElementType ET, el_o_t EO >
constexpr auto getSideNodeIndices(el_side_t side) -> std::span< const el_locind_t >
{
    if constexpr (Array_c< decltype(ElementTraits< Element< ET, EO > >::boundary_table) >)
        return {ElementTraits< Element< ET, EO > >::boundary_table.at(side)};
    else
        static_assert(util::always_false< ET >);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENTTRAITS_HPP
