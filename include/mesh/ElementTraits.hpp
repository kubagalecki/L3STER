#ifndef L3STER_MESH_ELEMENTTRAITS_HPP
#define L3STER_MESH_ELEMENTTRAITS_HPP

#include "mesh/ElementTypes.hpp"
#include "mesh/Point.hpp"

namespace lstr
{
template < ElementTypes ELTYPE, el_o_t ELORDER >
class Element;

template < typename >
struct ElementTraits;

template < el_o_t O >
struct ElementTraits< Element< ElementTypes::Hex, O > >
{
    static constexpr ElementTypes type              = ElementTypes::Hex;
    static constexpr el_o_t       order             = O;
    static constexpr n_id_t       nodes_per_element = (O + 1) * (O + 1) * (O + 1);
    static constexpr dim_t        native_dim        = 3;
    static constexpr el_ns_t      n_sides           = 6;

    using boundary_table_t = std::array< std::array< el_locind_t, (O + 1) * (O + 1) >, n_sides >;

private:
    static consteval boundary_table_t makeBoundaryTable()
    {
        boundary_table_t      bt{};
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
                bt[0][index] = index;                                   // front
                bt[1][index] = bt[0][index] + back_shift;               // back
                bt[2][index] = i * nodes_per_side + j;                  // bottom
                bt[3][index] = bt[2][index] + top_shift;                // top
                bt[4][index] = i * nodes_per_side + j * nodes_per_edge; // left
                bt[5][index] = bt[4][index] + right_shift;              // right
                ++index;
            }
        }
        return bt;
    }

public:
    static constexpr auto boundary_table = makeBoundaryTable();
};

template < el_o_t O >
struct ElementTraits< Element< ElementTypes::Quad, O > >
{
    static constexpr ElementTypes type              = ElementTypes::Quad;
    static constexpr el_o_t       order             = O;
    static constexpr n_id_t       nodes_per_element = (O + 1) * (O + 1);
    static constexpr dim_t        native_dim        = 2;
    static constexpr el_ns_t      n_sides           = 4;

    using boundary_table_t = std::array< std::array< el_locind_t, O + 1 >, n_sides >;

private:
    static consteval boundary_table_t makeBoundaryTable()
    {
        boundary_table_t      bt{};
        constexpr el_locind_t nodes_per_side = O + 1;
        constexpr auto        top_shift      = nodes_per_side * (nodes_per_side - 1);
        constexpr auto        right_shift    = nodes_per_side - 1;

        for (size_t i = 0; i < nodes_per_side; ++i)
        {
            bt[0][i] = i;                      // bottom
            bt[1][i] = bt[0][i] + top_shift;   // top
            bt[2][i] = i * nodes_per_side;     // left
            bt[3][i] = bt[2][i] + right_shift; // right
        }
        return bt;
    }

public:
    static constexpr auto boundary_table = makeBoundaryTable();
};

template < el_o_t O >
struct ElementTraits< Element< ElementTypes::Line, O > >
{
    static constexpr ElementTypes type              = ElementTypes::Line;
    static constexpr el_o_t       order             = O;
    static constexpr n_id_t       nodes_per_element = (O + 1);
    static constexpr dim_t        native_dim        = 1;
    static constexpr el_ns_t      n_sides           = 2;

    using boundary_table_t = std::array< std::array< el_locind_t, 1 >, n_sides >;

    static constexpr boundary_table_t boundary_table = {{{0}, {O}}};
};
} // namespace lstr
#endif // L3STER_MESH_ELEMENTTRAITS_HPP
