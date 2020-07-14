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
    static constexpr ElementTypes  element_type      = ElementTypes::Quad;
    static constexpr types::el_o_t element_order     = ELORDER;
    static constexpr types::n_id_t nodes_per_element = (ELORDER + 1) * (ELORDER + 1);
    static constexpr types::dim_t  native_dim        = 2;

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

    enum class Boundaries
    {
        Left,
        Right,
        Top,
        Bottom,
    };
};

template < types::el_o_t ELORDER >
struct ElementTraits< Element< ElementTypes::Line, ELORDER > >
{
    static constexpr ElementTypes  element_type      = ElementTypes::Line;
    static constexpr types::el_o_t element_order     = ELORDER;
    static constexpr types::n_id_t nodes_per_element = (ELORDER + 1);
    static constexpr types::dim_t  native_dim        = 1;

    struct ElementData
    {
        types::val_t L;
    };

    enum class Boundaries
    {
        Left,
        Right,
    };
};
} // namespace lstr::mesh

#endif // L3STER_MESH_ELEMENTTRAITS_HPP
