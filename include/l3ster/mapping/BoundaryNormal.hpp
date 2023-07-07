#ifndef L3STER_MAPPING_BOUNDARYNORMAL_HPP
#define L3STER_MAPPING_BOUNDARYNORMAL_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"

namespace lstr::map
{
template < ElementType ET, el_o_t EO >
auto computeBoundaryNormal(
    BoundaryElementView< ET, EO >                                                               el_view,
    const Eigen::Matrix< val_t, Element< ET, EO >::native_dim, Element< ET, EO >::native_dim >& jacobi_mat)
    -> Eigen::Vector< val_t, Element< ET, EO >::native_dim >
{
    Eigen::Vector< val_t, Element< ET, EO >::native_dim > retval;
    if constexpr (ET == ElementType::Line)
        retval[0] = el_view.getSide() == 0 ? -1. : 1.;
    else if constexpr (ET == ElementType::Quad)
    {
        switch (el_view.getSide())
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
    else if constexpr (ET == ElementType::Hex)
    {
        switch (el_view.getSide())
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
