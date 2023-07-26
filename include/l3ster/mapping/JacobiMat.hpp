#ifndef L3STER_MAPPING_JACOBIMAT_HPP
#define L3STER_MAPPING_JACOBIMAT_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/util/Algorithm.hpp"
#include "l3ster/util/Meta.hpp"

namespace lstr::map
{
// Jacobi matrix in native dimension, i.e. y=0, z=0 for 1D, z=0 for 2D is assumed
// Note: the generator returned from this function cannot outlive the element passed as its argument
template < mesh::ElementType T, el_o_t O >
    requires(T == mesh::ElementType::Line or T == mesh::ElementType::Quad or T == mesh::ElementType::Hex)
auto getNatJacobiMatGenerator(const mesh::Element< T, O >& element)
{
    return [&element](const Point< mesh::Element< T, O >::native_dim >& point) {
        constexpr auto nat_dim    = mesh::Element< T, O >::native_dim;
        constexpr auto n_o1_nodes = mesh::Element< T, 1 >::n_nodes;
        using ret_t               = Eigen::Matrix< val_t, nat_dim, nat_dim >;
        ret_t jac_mat             = ret_t::Zero();
        util::forConstexpr(
            [&]< size_t shapefun_ind >(std::integral_constant< size_t, shapefun_ind >) {
                util::forConstexpr(
                    [&]< dim_t derdim_ind >(std::integral_constant< dim_t, derdim_ind >) {
                        util::forConstexpr(
                            [&]< dim_t spacedim_ind >(std::integral_constant< dim_t, spacedim_ind >) {
                                const val_t vert_coord = element.getData().vertices[shapefun_ind][spacedim_ind];
                                const val_t shapefun_val =
                                    basis::ReferenceBasisFunction< T,
                                                                   1,
                                                                   shapefun_ind,
                                                                   basis::BasisType::Lagrange,
                                                                   basis::detail::derivativeByIndex(derdim_ind) >{}(
                                        point);
                                jac_mat(derdim_ind, spacedim_ind) += vert_coord * shapefun_val;
                            },
                            std::make_integer_sequence< dim_t, nat_dim >{});
                    },
                    std::make_integer_sequence< dim_t, nat_dim >{});
            },
            std::make_integer_sequence< size_t, n_o1_nodes >{});
        return jac_mat;
    };
}
} // namespace lstr::map
#endif // L3STER_MAPPING_JACOBIMAT_HPP
