#ifndef L3STER_MAPPING_BOUNDARYNORMAL_HPP
#define L3STER_MAPPING_BOUNDARYNORMAL_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"

namespace lstr::map
{
template < mesh::ElementType ET, el_o_t EO >
auto computeBoundaryNormal(
    el_side_t                                                                                               side,
    const Eigen::Matrix< val_t, mesh::Element< ET, EO >::native_dim, mesh::Element< ET, EO >::native_dim >& jacobi_mat)
    -> Eigen::Vector< val_t, mesh::Element< ET, EO >::native_dim >
{
    Eigen::Vector< val_t, mesh::Element< ET, EO >::native_dim > retval;
    if constexpr (ET == mesh::ElementType::Line)
        retval[0] = side == 0 ? -1. : 1.;
    else if constexpr (ET == mesh::ElementType::Quad)
    {
        switch (side)
        {
        case 0:
            retval[0] = jacobi_mat(0, 1);
            retval[1] = -jacobi_mat(0, 0);
            break;
        case 1:
            retval[0] = -jacobi_mat(0, 1);
            retval[1] = jacobi_mat(0, 0);
            break;
        case 2:
            retval[0] = -jacobi_mat(1, 1);
            retval[1] = jacobi_mat(1, 0);
            break;
        case 3:
            retval[0] = jacobi_mat(1, 1);
            retval[1] = -jacobi_mat(1, 0);
        }
        retval.normalize();
    }
    else if constexpr (ET == mesh::ElementType::Hex)
    {
        switch (side)
        {
        case 0:
            retval = -jacobi_mat.row(0).cross(jacobi_mat.row(1));
            break;
        case 1:
            retval = jacobi_mat.row(0).cross(jacobi_mat.row(1));
            break;
        case 2:
            retval = jacobi_mat.row(0).cross(jacobi_mat.row(2));
            break;
        case 3:
            retval = -jacobi_mat.row(0).cross(jacobi_mat.row(2));
            break;
        case 4:
            retval = -jacobi_mat.row(1).cross(jacobi_mat.row(2));
            break;
        case 5:
            retval = jacobi_mat.row(1).cross(jacobi_mat.row(2));
        }
        retval.normalize();
    }
    return retval;
}
} // namespace lstr::map
#endif // L3STER_MAPPING_BOUNDARYNORMAL_HPP
