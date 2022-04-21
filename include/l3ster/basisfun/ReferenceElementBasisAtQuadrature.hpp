#ifndef L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP

#include "l3ster/basisfun/ComputeRefBasisAtQpoints.hpp"
#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"

namespace lstr
{
template < BasisTypes BT, ElementTypes ET, el_o_t EO, QuadratureTypes QT, q_o_t QO >
const auto& getReferenceBasisAtDomainQuadrature()
{
    static const auto value = [] {
        const auto quadrature = getQuadrature< QT, QO, ET >();
        const auto basis_vals = computeRefBasisAtQpoints< BT, ET, EO >(quadrature);
        const auto basis_ders = computeRefBasisDersAtQpoints< BT, ET, EO >(quadrature);
        return ReferenceBasisAtQuadrature< ET, EO, quadrature.size, quadrature.dim >{
            .quadrature = quadrature, .basis_vals = basis_vals, .basis_ders = basis_ders};
    }();
    return value;
}
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEELEMENTBASISATQUADRATURE_HPP
