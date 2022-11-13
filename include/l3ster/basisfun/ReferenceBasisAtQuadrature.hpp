#ifndef L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
#define L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP

#include "l3ster/basisfun/ReferenceBasisAtPoints.hpp"
#include "l3ster/quad/GenerateQuadrature.hpp"

namespace lstr
{
template < ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
struct ReferenceBasisAtQuadrature
{
    Quadrature< QL, QD >                 quadrature;
    ReferenceBasisAtPoints< ET, EO, QL > basis;
};
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISATQUADRATURE_HPP
