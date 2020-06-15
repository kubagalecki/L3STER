// ElementTraits class implementation

#ifndef L3STER_INCGUARD_MESH_ELEMENTTRAITS_HPP
#define L3STER_INCGUARD_MESH_ELEMENTTRAITS_HPP

#include "mesh/ElementTypes.hpp"
#include "typedefs/Types.h"

namespace lstr::mesh
{

// Forward-declare the Element class
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
    static constexpr ElementTypes  element_type = ElementTypes::Quad;
    static constexpr types::el_o_t element_order = ELORDER;

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

} // namespace lstr::mesh

#endif // end include guard
