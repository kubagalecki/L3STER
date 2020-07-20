#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

namespace lstr::mesh
{
class BoundaryView
{
public:
    template < ElementTypes ELTYPE, types::el_o_t ELORDER >
    using boundary_element_view_vector_t =
        std::vector< BoundaryElementView< Element< ELTYPE, ELORDER > > >;
    using boundary_element_view_vector_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant,
                                                          boundary_element_view_vector_t >;
    using boundary_element_view_vector_variant_vector_t =
        std::vector< boundary_element_view_vector_variant_t >;

    BoundaryView()                    = delete;
    BoundaryView(const BoundaryView&) = default;
    BoundaryView(BoundaryView&&)      = default;
    BoundaryView& operator=(const BoundaryView&) = delete;
    BoundaryView& operator=(BoundaryView&&) = delete;
    inline BoundaryView(const MeshPartition& mesh, const types::d_id_t& boundary_id);

private:
    boundary_element_view_vector_variant_vector_t boundary_element_view_vectors;
};

inline BoundaryView::BoundaryView(const MeshPartition& mesh, const types::d_id_t& boundary_id)
{}
} // namespace lstr::mesh

#endif // L3STER_MESH_BOUNDARYVIEW_HPP
