#ifndef L3STER_ASSEMBLY_REFERENCEBASISATQUADRATURE_HPP
#define L3STER_ASSEMBLY_REFERENCEBASISATQUADRATURE_HPP

#include "l3ster/mesh/ElementTraits.hpp"
#include "l3ster/quad/GenerateQuadrature.hpp"

namespace lstr
{
template < ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
struct ReferenceBasisAtQuadrature
{
private:
    using basis_at_qp_t = Eigen::Matrix< val_t, QL, ElementTraits< Element< ET, EO > >::nodes_per_element >;
    using basis_ders_t  = std::array< basis_at_qp_t, ElementTraits< Element< ET, EO > >::native_dim >;

public:
    Quadrature< QL, QD > quadrature;
    basis_at_qp_t        basis_vals;
    basis_ders_t         basis_ders;
};

template < BasisTypes BT, ElementTypes ET, el_o_t EO, QuadratureTypes QT, q_o_t QO >
const auto& getReferenceBasisAtDomainQuadrature()
{
    static const auto value = [] {
        const auto quadrature = getQuadrature< QT, QO, ET >();
        const auto basis_vals = computeRefBasesAtQpoints< BT, ET, EO >(quadrature);
        const auto basis_ders = computeRefBasisDersAtQpoints< BT, ET, EO >(quadrature);
        return ReferenceBasisAtQuadrature< ET, EO, quadrature.size, quadrature.dim >{
            .quadrature = quadrature, .basis_vals = basis_vals, .basis_ders = basis_ders};
    }();
    return value;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_REFERENCEBASISATQUADRATURE_HPP
