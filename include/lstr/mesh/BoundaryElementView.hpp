#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "lstr/mesh/Aliases.hpp"
#include "lstr/mesh/Element.hpp"

namespace lstr::mesh
{
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
struct BoundaryElementView
{
    using element_t = Element< ELTYPE, ELORDER >;

    BoundaryElementView(const element_t& element_, const types::el_ns_t element_side_)
        : element{std::cref(element_)}, element_side{element_side_}
    {}

    element_cref_t< ELTYPE, ELORDER > element;
    types::el_ns_t                    element_side;
};
} // namespace lstr::mesh

#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
