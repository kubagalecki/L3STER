#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

namespace lstr::mesh
{
template < typename >
struct BoundaryElementView;

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
struct BoundaryElementView< Element< ELTYPE, ELORDER > >
{
    using element_t  = Element< ELTYPE, ELORDER >;
    using boundary_t = typename ElementTraits< element_t >::Boundaries;

    const element_t& element;
    boundary_t       boundary;
};
} // namespace lstr::mesh

#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
