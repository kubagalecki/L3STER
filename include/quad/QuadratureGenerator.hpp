// Quadrature generation functionality

#ifndef L3STER_QUAD_QUADRATUREGENERATOR_HPP
#define L3STER_QUAD_QUADRATUREGENERATOR_HPP

#include "mesh/Element.hpp"
#include "quad/ReferenceQuadrature.hpp"

namespace lstr::quad
{
template < QuadratureTypes QTYPE, types::q_o_t QORDER >
class QuadratureGenerator
{
public:
    template < types::el_o_t ELORDER >
    [[nodiscard]] const auto& get(const mesh::Element< mesh::ElementTypes::Line, ELORDER >&) const;

    template < types::el_o_t ELORDER >
    [[nodiscard]] const auto& get(const mesh::Element< mesh::ElementTypes::Quad, ELORDER >&) const;
};

template < QuadratureTypes QTYPE, types::q_o_t QORDER >
template < types::el_o_t ELORDER >
const auto& QuadratureGenerator< QTYPE, QORDER >::get(const mesh::Element< mesh::ElementTypes::Line, ELORDER >&) const
{
    return ReferenceQuadrature< QTYPE, QORDER >::value;
}

template < QuadratureTypes QTYPE, types::q_o_t QORDER >
template < types::el_o_t ELORDER >
const auto& QuadratureGenerator< QTYPE, QORDER >::get(const mesh::Element< mesh::ElementTypes::Quad, ELORDER >&) const
{
    static const auto quad = [] {
        using ref_quadrature_t  = ReferenceQuadrature< QTYPE, QORDER >;
        constexpr auto ref_size = ReferenceQuadratureTraits< ref_quadrature_t >::size;
        using quadrature_t      = Quadrature< ref_size * ref_size, 2 >;

        const auto& ref_quadrature   = ref_quadrature_t::value;
        const auto& ref_quadrature_p = ref_quadrature.getQPoints();
        const auto& ref_quadrature_w = ref_quadrature.getWeights();

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;

        for (size_t i = 0; i < ref_size; ++i)
        {
            for (size_t j = 0; j < ref_size; ++j)
            {
                q_points[i * ref_size + j][0] = ref_quadrature_p[i][0];
                q_points[i * ref_size + j][1] = ref_quadrature_p[j][0];
                weights[i * ref_size + j]     = ref_quadrature_w[i] * ref_quadrature_w[j];
            }
        }

        return quadrature_t{q_points, weights};
    }();

    return quad;
}
} // namespace lstr::quad

#endif // L3STER_QUAD_QUADRATUREGENERATOR_HPP
