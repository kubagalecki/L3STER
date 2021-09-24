#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "Aliases.hpp"
#include "Element.hpp"

namespace lstr
{
template < ElementTypes ELTYPE, el_o_t ELORDER >
struct BoundaryElementView
{
    using element_t = Element< ELTYPE, ELORDER >;

    BoundaryElementView(const element_t& element_, const el_ns_t element_side_)
        : element{&element_}, element_side{element_side_}
    {}

    element_cptr_t< ELTYPE, ELORDER > element;
    el_ns_t                           element_side;
};
} // namespace lstr

#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
