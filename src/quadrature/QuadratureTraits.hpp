#ifndef L3STER_QUADRATURE_QUADRATURETRAITS_HPP
#define L3STER_QUADRATURE_QUADRATURETRAITS_HPP

namespace lstr::quad
{
template < typename Element, QuadratureTypes, types::q_o_t >
struct QuadratureTraits;

template < types::el_o_t ELORDER, types::q_o_t QORDER >
struct QuadratureTraits< mesh::Element< mesh::ElementTypes::Quad, ELORDER >,
                         QuadratureTypes::GLeg,
                         QORDER >
{
    static constexpr types::q_l_t size = (QORDER / 2 + 1) * (QORDER / 2 + 1);
};

template < types::el_o_t ELORDER, types::q_o_t QORDER >
struct QuadratureTraits< mesh::Element< mesh::ElementTypes::Line, ELORDER >,
                         QuadratureTypes::GLeg,
                         QORDER >
{
    static constexpr types::q_l_t size = QORDER / 2 + 1;
};

template < types::el_o_t ELORDER, types::q_o_t QORDER >
struct QuadratureTraits< mesh::Element< mesh::ElementTypes::Quad, ELORDER >,
                         QuadratureTypes::GLob,
                         QORDER >
{
    static constexpr types::q_l_t size = (QORDER / 2 + 3) * (QORDER / 2 + 3);
};

template < types::el_o_t ELORDER, types::q_o_t QORDER >
struct QuadratureTraits< mesh::Element< mesh::ElementTypes::Line, ELORDER >,
                         QuadratureTypes::GLob,
                         QORDER >
{
    static constexpr types::q_l_t size = QORDER / 2 + 3;
};

} // namespace lstr::quad

#endif // L3STER_QUADRATURE_QUADRATURETRAITS_HPP
