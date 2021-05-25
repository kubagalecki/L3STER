#ifndef L3STER_MAPPING_JACOBIMAT_HPP
#define L3STER_MAPPING_JACOBIMAT_HPP

#include "basisfun/ReferenceBasisFunction.hpp"

#include "Eigen/Core"

namespace lstr
{
// Jacobi matrix in native dimension, i.e. y=0, z=0 for 1D, z=0 for 2D is assumed
template < ElementTypes T, el_o_t O >
auto getNatJacobiMatGenerator(const Element< T, O >& el)
{
    using ret_t = Eigen::
        Matrix< val_t, ElementTraits< Element< T, O > >::native_dim, ElementTraits< Element< T, O > >::native_dim >;
    if constexpr (T == ElementTypes::Line)
    {
        return [val = (el.getData().vertices[1].x() - el.getData().vertices[0].x()) / 2.](const Point< 1 >&) {
            return ret_t{val};
        };
    }
    else if constexpr (T == ElementTypes::Quad)
    {
        return [&](const Point< 2 >& ref_point) {
            ret_t jac_mat = ret_t::Zero();
            [&]< size_t... I >(std::index_sequence< I... >)
            {
                const auto contrib_i = [&]< size_t Ind >(std::integral_constant< size_t, Ind >) {
                    jac_mat(0, 0) +=
                        ReferenceBasisFunctionDerXi< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].x();
                    jac_mat(0, 1) +=
                        ReferenceBasisFunctionDerXi< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].y();
                    jac_mat(1, 0) +=
                        ReferenceBasisFunctionDerEta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].x();
                    jac_mat(1, 1) +=
                        ReferenceBasisFunctionDerEta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].y();
                };
                (contrib_i(std::integral_constant< size_t, I >{}), ...);
            }
            (std::make_index_sequence< 4 >{});
            return jac_mat;
        };
    }
    else if constexpr (T == ElementTypes::Hex)
    {
        return [&](const Point< 3 >& ref_point) {
            ret_t jac_mat = ret_t::Zero();
            [&]< size_t... I >(std::index_sequence< I... >)
            {
                const auto contrib_i = [&]< size_t Ind >(std::integral_constant< size_t, Ind >) {
                    jac_mat(0, 0) +=
                        ReferenceBasisFunctionDerXi< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].x();
                    jac_mat(0, 1) +=
                        ReferenceBasisFunctionDerXi< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].y();
                    jac_mat(0, 2) +=
                        ReferenceBasisFunctionDerXi< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].z();
                    jac_mat(1, 0) +=
                        ReferenceBasisFunctionDerEta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].x();
                    jac_mat(1, 1) +=
                        ReferenceBasisFunctionDerEta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].y();
                    jac_mat(1, 2) +=
                        ReferenceBasisFunctionDerEta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].z();
                    jac_mat(2, 0) +=
                        ReferenceBasisFunctionDerZeta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].x();
                    jac_mat(2, 1) +=
                        ReferenceBasisFunctionDerZeta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].y();
                    jac_mat(2, 2) +=
                        ReferenceBasisFunctionDerZeta< T, 1, Ind >{}(ref_point)*el.getData().vertices[Ind].z();
                };
                (contrib_i(std::integral_constant< size_t, I >{}), ...);
            }
            (std::make_index_sequence< 8 >{});
            return jac_mat;
        };
    }
}
} // namespace lstr
#endif // L3STER_MAPPING_JACOBIMAT_HPP
