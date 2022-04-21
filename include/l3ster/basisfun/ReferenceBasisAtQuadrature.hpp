#ifndef L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP

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
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
