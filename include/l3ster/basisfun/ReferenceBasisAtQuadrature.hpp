#ifndef L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP

#include "l3ster/basisfun/ReferenceBasisAtPoints.hpp"
#include "l3ster/quad/GenerateQuadrature.hpp"

#include <ranges>

namespace lstr::basis
{
template < mesh::ElementType ET, el_o_t EO, q_l_t QL >
struct ReferenceBasisAtQuadrature
{
    template < typename Fun >
    void forEach(Fun&& fun) const
    {
        for (auto&& [quad_point, quad_wgt, basis_val, basis_der] :
             std::views::zip(quadrature.points, quadrature.weights, basis.values, basis.derivatives))
            std::invoke(fun, quad_point, quad_wgt, basis_val, basis_der);
    }

    quad::Quadrature< QL, mesh::Element< ET, EO >::native_dim > quadrature;
    ReferenceBasisAtPoints< ET, EO, QL >                        basis;
};
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
