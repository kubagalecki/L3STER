#ifndef L3STER_MAPPING_JACOBIMAT_HPP
#define L3STER_MAPPING_JACOBIMAT_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Meta.hpp"

namespace lstr::map
{
// Jacobi matrix in native dimension: assumes y=0, z=0 for 1D, z=0 for 2D
template < mesh::ElementType ET >
using JacobiMat = Eigen::Matrix< val_t, mesh::Element< ET, 1 >::native_dim, mesh::Element< ET, 1 >::native_dim >;

// Note: the generator returned from this function cannot outlive the element data passed as its argument
template < mesh::ElementType T, el_o_t O >
    requires(T == mesh::ElementType::Line or T == mesh::ElementType::Quad or T == mesh::ElementType::Hex)
auto getNatJacobiMatGenerator(const mesh::ElementData< T, O >& element_data)
{
    constexpr auto n_vert_comps = mesh::ElementData< T, O >::vertex_array_t::value_type::dimension;
    constexpr auto nat_dim      = mesh::Element< T, O >::native_dim;
    constexpr auto n_o1_nodes   = mesh::Element< T, 1 >::n_nodes;
    using vert_tab_t            = Eigen::Matrix< val_t, n_vert_comps, n_o1_nodes >;
    return [&](const Point< mesh::Element< T, O >::native_dim >& point) -> JacobiMat< T > {
        const auto shape_ders = basis::computeReferenceBasisDerivatives< T, 1, basis::BasisType::Lagrange >(point);
        const auto verts_raw  = element_data.vertices.front().coords.data();
        const auto vert_tab   = Eigen::Map< const vert_tab_t >{verts_raw};
        auto       retval     = JacobiMat< T >{shape_ders * vert_tab.template topRows< nat_dim >().transpose()};
        return retval;
    };
}
} // namespace lstr::map
#endif // L3STER_MAPPING_JACOBIMAT_HPP
