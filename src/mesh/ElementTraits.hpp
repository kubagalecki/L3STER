#ifndef L3STER_MESH_ELEMENTTRAITS_HPP
#define L3STER_MESH_ELEMENTTRAITS_HPP

namespace lstr::mesh
{
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element;

template < typename Element >
struct ElementTraits;

template < types::el_o_t ELORDER >
struct ElementTraits< Element< ElementTypes::Quad, ELORDER > >
{
    static constexpr ElementTypes       element_type       = ElementTypes::Quad;
    static constexpr types::el_o_t      element_order      = ELORDER;
    static constexpr types::n_id_t      nodes_per_element  = (ELORDER + 1) * (ELORDER + 1);
    static constexpr types::dim_t       native_dim         = 2;
    static constexpr types::el_ns_t     n_sides            = 4;
    static constexpr types::el_locind_t max_nodes_per_side = ELORDER + 1;

    using boundary_table_t =
        std::array< std::array< types::el_locind_t, max_nodes_per_side + 1 >, n_sides >;

private:
    static constexpr boundary_table_t makeBoundaryTable()
    {
        boundary_table_t bt{};

        for (size_t i = 0; i < n_sides; ++i)
            bt[i][0] = max_nodes_per_side;

        for (size_t i = 0; i < max_nodes_per_side; ++i)
        {
            bt[0][i + 1] = i;                                                 // bottom
            bt[1][i + 1] = i * max_nodes_per_side;                            // left
            bt[2][i + 1] = i + (max_nodes_per_side - 1) * max_nodes_per_side; // top
            bt[3][i + 1] = i * max_nodes_per_side + max_nodes_per_side - 1;   // right
        }

        return bt;
    }

public:
    static constexpr boundary_table_t boundary_table = makeBoundaryTable();

    struct ElementData
    {
        types::val_t a;
        types::val_t b;
        types::val_t c;
        types::val_t alphax;
        types::val_t alphay;
        types::val_t betax;
        types::val_t betay;
        types::val_t gammax;
        types::val_t gammay;
    };
};

template < types::el_o_t ELORDER >
struct ElementTraits< Element< ElementTypes::Line, ELORDER > >
{
    static constexpr ElementTypes       element_type       = ElementTypes::Line;
    static constexpr types::el_o_t      element_order      = ELORDER;
    static constexpr types::n_id_t      nodes_per_element  = (ELORDER + 1);
    static constexpr types::dim_t       native_dim         = 1;
    static constexpr types::el_ns_t     n_sides            = 2;
    static constexpr types::el_locind_t max_nodes_per_side = 1;

    using boundary_table_t =
        std::array< std::array< types::el_locind_t, max_nodes_per_side + 1 >, n_sides >;

    static constexpr boundary_table_t boundary_table = {{{1, 0}, {1, ELORDER}}};

    struct ElementData
    {
        types::val_t L;
    };
};
} // namespace lstr::mesh

#endif // L3STER_MESH_ELEMENTTRAITS_HPP
