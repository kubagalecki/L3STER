#ifndef L3STER_MESH_ELEMENTTRAITS_HPP
#define L3STER_MESH_ELEMENTTRAITS_HPP

#include "l3ster/common/Structs.hpp"
#include "l3ster/mesh/ElementType.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/ConstexprVector.hpp"

#include <vector>

namespace lstr::mesh
{
template < ElementType ELTYPE, el_o_t ELORDER >
class Element;

template < typename >
struct ElementTraits;

namespace detail::elem
{
consteval auto getBoundaryInds(const auto& boundary_table) -> util::ConstexprVector< el_locind_t >
{
    auto retval = util::ConstexprVector< el_locind_t >{};
    util::forEachTuple(boundary_table, [&retval](const auto& side_boundary_inds) {
        for (auto i : side_boundary_inds)
            retval.pushBack(i);
    });
    std::ranges::sort(retval);
    const auto erase_size = std::ranges::size(std::ranges::unique(retval));
    for (size_t i = 0; i < erase_size; ++i)
        retval.popBack();
    return retval;
}

template < typename BoundaryTable >
consteval auto makeBoundaryNodeInds()
{
    constexpr auto   boundary_table = BoundaryTable::compute();
    constexpr size_t size           = getBoundaryInds(boundary_table).size();
    auto             retval         = std::array< el_locind_t, size >{};
    std::ranges::copy(getBoundaryInds(boundary_table), begin(retval));
    return retval;
}

template < typename BoundaryTable, el_locind_t nodes_per_element >
consteval auto makeInternalNodeInds()
{
    constexpr auto boundary_table   = BoundaryTable::compute();
    constexpr auto n_boundary_nodes = getBoundaryInds(boundary_table).size();
    static_assert(nodes_per_element >= n_boundary_nodes);
    constexpr size_t size          = nodes_per_element - n_boundary_nodes;
    const auto       boundary_inds = getBoundaryInds(boundary_table);
    auto             retval        = std::array< el_locind_t, size >{};
    std::ranges::copy(std::views::iota(el_locind_t{0}, nodes_per_element) |
                          std::views::filter([&](auto i) { return not std::ranges::binary_search(boundary_inds, i); }),
                      begin(retval));
    return retval;
}
} // namespace detail::elem

template < el_o_t O >
struct ElementTraits< Element< ElementType::Hex, O > >
{
    static constexpr ElementType type              = ElementType::Hex;
    static constexpr el_o_t      order             = O;
    static constexpr el_locind_t nodes_per_element = (O + 1) * (O + 1) * (O + 1);
    static constexpr dim_t       native_dim        = 3;
    static constexpr el_side_t   n_sides           = 6;

private:
    struct BoundaryTable
    {
        static consteval auto compute()
        {
            auto                  retval = std::array< std::array< el_locind_t, (O + 1) * (O + 1) >, n_sides >{};
            constexpr el_locind_t nodes_per_edge = O + 1;
            constexpr auto        nodes_per_side = nodes_per_edge * nodes_per_edge;
            constexpr auto        back_shift     = nodes_per_side * (nodes_per_edge - 1);
            constexpr auto        top_shift      = nodes_per_edge * (nodes_per_edge - 1);
            constexpr auto        right_shift    = nodes_per_edge - 1;

            size_t index = 0;
            for (size_t i = 0; i < nodes_per_edge; ++i)
            {
                for (size_t j = 0; j < nodes_per_edge; ++j)
                {
                    retval[0][index] = static_cast< el_locind_t >(index);                                   // front
                    retval[1][index] = static_cast< el_locind_t >(retval[0][index] + back_shift);           // back
                    retval[2][index] = static_cast< el_locind_t >(i * nodes_per_side + j);                  // bottom
                    retval[3][index] = static_cast< el_locind_t >(retval[2][index] + top_shift);            // top
                    retval[4][index] = static_cast< el_locind_t >(i * nodes_per_side + j * nodes_per_edge); // left
                    retval[5][index] = static_cast< el_locind_t >(retval[4][index] + right_shift);          // right
                    ++index;
                }
            }
            return retval;
        }
    };

public:
    static constexpr auto boundary_table     = BoundaryTable::compute();
    static constexpr auto boundary_node_inds = detail::elem::makeBoundaryNodeInds< BoundaryTable >();
    static constexpr auto internal_node_inds = detail::elem::makeInternalNodeInds< BoundaryTable, nodes_per_element >();
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Quad, O > >
{
    static constexpr ElementType type              = ElementType::Quad;
    static constexpr el_o_t      order             = O;
    static constexpr el_locind_t nodes_per_element = (O + 1) * (O + 1);
    static constexpr dim_t       native_dim        = 2;
    static constexpr el_side_t   n_sides           = 4;

private:
    struct BoundaryTable
    {
        static consteval auto compute()
        {
            std::array< std::array< el_locind_t, O + 1 >, n_sides > retval{};
            constexpr el_locind_t                                   nodes_per_side = O + 1;
            constexpr auto                                          top_shift   = nodes_per_side * (nodes_per_side - 1);
            constexpr auto                                          right_shift = nodes_per_side - 1;

            for (size_t i = 0; i < nodes_per_side; ++i)
            {
                retval[0][i] = static_cast< el_locind_t >(i);                          // bottom
                retval[1][i] = static_cast< el_locind_t >(retval[0][i] + top_shift);   // top
                retval[2][i] = static_cast< el_locind_t >(i * nodes_per_side);         // left
                retval[3][i] = static_cast< el_locind_t >(retval[2][i] + right_shift); // right
            }
            return retval;
        }
    };

public:
    static constexpr auto boundary_table     = BoundaryTable::compute();
    static constexpr auto boundary_node_inds = detail::elem::makeBoundaryNodeInds< BoundaryTable >();
    static constexpr auto internal_node_inds = detail::elem::makeInternalNodeInds< BoundaryTable, nodes_per_element >();
};

template < el_o_t O >
struct ElementTraits< Element< ElementType::Line, O > >
{
    static constexpr ElementType type              = ElementType::Line;
    static constexpr el_o_t      order             = O;
    static constexpr el_locind_t nodes_per_element = (O + 1);
    static constexpr dim_t       native_dim        = 1;
    static constexpr el_side_t   n_sides           = 2;

private:
    struct BoundaryTable
    {
        static consteval auto compute() -> std::array< std::array< el_locind_t, 1 >, n_sides > { return {{{0}, {O}}}; }
    };

public:
    static constexpr auto boundary_table     = BoundaryTable::compute();
    static constexpr auto boundary_node_inds = detail::elem::makeBoundaryNodeInds< BoundaryTable >();
    static constexpr auto internal_node_inds = detail::elem::makeInternalNodeInds< BoundaryTable, nodes_per_element >();
};
} // namespace lstr::mesh
#endif // L3STER_MESH_ELEMENTTRAITS_HPP
