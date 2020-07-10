#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP

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

template < types::q_o_t QORDER >
struct ReferenceQuadratureTraits< ReferenceQuadrature< QuadratureTypes::GLob, QORDER > >
{
    static constexpr types::q_l_t size = QORDER / 2 + 3;
};
} // namespace lstr::quad

#endif // L3STER_QUADRATURE_REFERENCEQUADRATURETRAITS_HPP
