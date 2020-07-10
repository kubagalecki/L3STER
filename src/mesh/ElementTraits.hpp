#ifndef L3STER_INCGUARD_MESH_ELEMENTTRAITS_HPP
#define L3STER_INCGUARD_MESH_ELEMENTTRAITS_HPP

namespace lstr::mesh
{
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element;

//////////////////////////////////////////////////////////////////////////////////////////////
//                                   ELEMENT TRAITS CLASS                                   //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
This class is to be specialized for each element type and/or order. It must contain the member
class ElementData, which holds useful physical-element-specific information (e.g. for
computing the Jacobian)
*/
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
        double a;
        double b;
        double c;
        double alphax;
        double alphay;
        double betax;
        double betay;
        double gammax;
        double gammay;
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
        ElementTypes   parent_el_t;
        types::el_id_t parent_el;
        types::el_f_t  face;
    };
};
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_ELEMENTTRAITS_HPP
