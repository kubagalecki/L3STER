#ifndef L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP

#include "l3ster/basisfun/ReferenceBasisAtPoints.hpp"
#include "l3ster/quad/GenerateQuadrature.hpp"

namespace lstr::basis
{
template < ElementTypes ET, el_o_t EO, q_l_t QL >
struct ReferenceBasisAtQuadrature
{
    quad::Quadrature< QL, Element< ET, EO >::native_dim > quadrature;
    ReferenceBasisAtPoints< ET, EO, QL >                  basis;
};
} // namespace lstr::basis
#endif // L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
