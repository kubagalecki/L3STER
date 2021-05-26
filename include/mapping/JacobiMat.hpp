#ifndef L3STER_MAPPING_JACOBIMAT_HPP
#define L3STER_MAPPING_JACOBIMAT_HPP

#include "basisfun/ReferenceBasisFunction.hpp"
#include "util/Algorithm.hpp"
#include "util/Meta.hpp"

#include "Eigen/Core"

namespace lstr
{
// Jacobi matrix in native dimension, i.e. y=0, z=0 for 1D, z=0 for 2D is assumed
template < ElementTypes T, el_o_t O >
auto getNatJacobiMatGenerator(const Element< T, O >& el)
{
    if constexpr (T == ElementTypes::Line or T == ElementTypes::Quad or T == ElementTypes::Hex)
    {
        return [&](const Point< ElementTraits< Element< T, O > >::native_dim >& point) {
            constexpr auto nat_dim    = ElementTraits< Element< T, O > >::native_dim;
            constexpr auto n_o1_nodes = Element< T, 1 >::n_nodes;
            using ret_t               = Eigen::Matrix< val_t, nat_dim, nat_dim >;
            ret_t jac_mat             = ret_t::Zero();
            forConstexpr(
                [&]< size_t SHAPEF >(std::integral_constant< size_t, SHAPEF >) {
                    forConstexpr(
                        [&]< size_t DERDIM >(std::integral_constant< size_t, DERDIM >) {
                            forConstexpr(
                                [&]< dim_t SDIM >(std::integral_constant< dim_t, SDIM >) {
                                    const val_t vert_coord   = el.getData().vertices[SHAPEF][SDIM];
                                    const val_t shapefun_val = ReferenceBasisFunction< T, 1, SHAPEF, DERDIM >{}(point);
                                    jac_mat(DERDIM - 1, SDIM) += vert_coord * shapefun_val;
                                },
                                std::make_integer_sequence< dim_t, nat_dim >{});
                        },
                        int_seq_interval< size_t, 1, nat_dim >{});
                },
                std::make_integer_sequence< size_t, n_o1_nodes >{});
            return jac_mat;
        };
    }
}
} // namespace lstr
#endif // L3STER_MAPPING_JACOBIMAT_HPP
