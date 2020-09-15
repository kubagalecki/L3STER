#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "lstr/mesh/Element.hpp"
#include "lstr/mesh/Aliases.hpp"

namespace lstr::mesh
{
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
struct BoundaryElementView
{
    using element_t = Element< ELTYPE, ELORDER >;

    BoundaryElementView()                               = delete;
    BoundaryElementView(const BoundaryElementView&)     = default;
    BoundaryElementView(BoundaryElementView&&) noexcept = default;
    BoundaryElementView& operator=(const BoundaryElementView&) = default;
    BoundaryElementView& operator=(BoundaryElementView&&) noexcept = default;
    BoundaryElementView(const element_t& element_, const types::el_ns_t element_side_)
        : element{std::cref(element_)}, element_side{element_side_}
    {}

    element_cref_t< ELTYPE, ELORDER > element;
    types::el_ns_t                    element_side;
};
} // namespace lstr::mesh

#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
