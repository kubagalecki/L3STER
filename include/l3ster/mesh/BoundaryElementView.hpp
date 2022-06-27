#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "l3ster/mesh/Aliases.hpp"
#include "l3ster/mesh/Element.hpp"

namespace lstr
{
template < ElementTypes ELTYPE, el_o_t ELORDER >
struct BoundaryElementView
{
    using element_t = Element< ELTYPE, ELORDER >;

    BoundaryElementView(const element_t& element_, const el_side_t element_side_)
        : element{&element_}, element_side{element_side_}
    {}

    element_cptr_t< ELTYPE, ELORDER > element;
    el_side_t                         element_side;
};
} // namespace lstr
#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
