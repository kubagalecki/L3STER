#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP

#include "QuadratureTypes.h"
#include "l3ster/defs/Typedefs.h"

namespace lstr
{
template < QuadratureTypes, q_o_t >
struct ReferenceQuadrature;

template < typename >
struct ReferenceQuadratureTraits;

template < q_o_t QORDER >
struct ReferenceQuadratureTraits< ReferenceQuadrature< QuadratureTypes::GLeg, QORDER > >
{
    static constexpr q_l_t size = QORDER / 2 + 1;
};
} // namespace lstr

#endif // L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP
