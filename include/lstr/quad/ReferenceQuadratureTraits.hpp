#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP

#include "lstr/defs/Typedefs.h"
#include "lstr/quad/QuadratureTypes.h"

namespace lstr::quad
{
template < QuadratureTypes, types::q_o_t >
struct ReferenceQuadrature;

template < typename >
struct ReferenceQuadratureTraits;

template < types::q_o_t QORDER >
struct ReferenceQuadratureTraits< ReferenceQuadrature< QuadratureTypes::GLeg, QORDER > >
{
    static constexpr types::q_l_t size = QORDER / 2 + 1;
};
} // namespace lstr::quad

#endif // L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP
