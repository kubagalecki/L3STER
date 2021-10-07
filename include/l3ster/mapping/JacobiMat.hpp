#ifndef L3STER_MAPPING_JACOBIMAT_HPP
#define L3STER_MAPPING_JACOBIMAT_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Meta.hpp"

#include "Eigen/Core"

namespace lstr
{
// Jacobi matrix in native dimension, i.e. y=0, z=0 for 1D, z=0 for 2D is assumed
template < ElementTypes T, el_o_t O >
requires(T == ElementTypes::Line or T == ElementTypes::Quad or
         T == ElementTypes::Hex) auto getNatJacobiMatGenerator(const Element< T, O >& el)
{
    return [&](const Point< ElementTraits< Element< T, O > >::native_dim >& point) {
        constexpr auto nat_dim    = ElementTraits< Element< T, O > >::native_dim;
        constexpr auto n_o1_nodes = Element< T, 1 >::n_nodes;
        using ret_t               = Eigen::Matrix< val_t, nat_dim, nat_dim >;
        ret_t jac_mat             = ret_t::Zero();
        forConstexpr(
            [&]< size_t SHAPEFUN_IND >(std::integral_constant< size_t, SHAPEFUN_IND >) {
                forConstexpr(
                    [&]< dim_t DERDIM_IND >(std::integral_constant< dim_t, DERDIM_IND >) {
                        forConstexpr(
                            [&]< dim_t SPACEDIM_IND >(std::integral_constant< dim_t, SPACEDIM_IND >) {
                                const val_t vert_coord = el.getData().vertices[SHAPEFUN_IND][SPACEDIM_IND];
                                const val_t shapefun_val =
                                    ReferenceBasisFunction< T,
                                                            1, // Lagrange 1 == shape fun
                                                            SHAPEFUN_IND,
                                                            BasisTypes::Lagrange, // Lagrange 1 == shape fun
                                                            detail::derivativeByIndex(DERDIM_IND) >{}(point);
                                jac_mat(DERDIM_IND, SPACEDIM_IND) += vert_coord * shapefun_val;
                            },
                            std::make_integer_sequence< dim_t, nat_dim >{});
                    },
                    std::make_integer_sequence< dim_t, nat_dim >{});
            },
            std::make_integer_sequence< size_t, n_o1_nodes >{});
        return jac_mat;
    };
}
} // namespace lstr
#endif // L3STER_MAPPING_JACOBIMAT_HPP
