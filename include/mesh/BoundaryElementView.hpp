#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "mesh/Aliases.hpp"
#include "mesh/Element.hpp"

namespace lstr
{
template < ElementTypes ELTYPE, el_o_t ELORDER >
struct BoundaryElementView
{
    using element_t = Element< ELTYPE, ELORDER >;

    BoundaryElementView(const element_t& element_, const el_ns_t element_side_)
        : element{std::cref(element_)}, element_side{element_side_}
    {}

    element_cref_t< ELTYPE, ELORDER > element;
    el_ns_t                           element_side;
};
} // namespace lstr

#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
