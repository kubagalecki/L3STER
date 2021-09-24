// Quadrature generation functionality

#ifndef L3STER_QUAD_QUADRATUREGENERATOR_HPP
#define L3STER_QUAD_QUADRATUREGENERATOR_HPP

#include "ReferenceQuadrature.hpp"
#include "l3ster/mesh/Element.hpp"

namespace lstr
{
template < QuadratureTypes QTYPE, q_o_t QORDER >
struct QuadratureGenerator
{
    template < el_o_t ELORDER >
    [[nodiscard]] const auto& get(const Element< ElementTypes::Line, ELORDER >&) const;

    template < el_o_t ELORDER >
    [[nodiscard]] const auto& get(const Element< ElementTypes::Quad, ELORDER >&) const;

    template < el_o_t ELORDER >
    [[nodiscard]] const auto& get(const Element< ElementTypes::Hex, ELORDER >&) const;
};

template < QuadratureTypes QTYPE, q_o_t QORDER >
template < el_o_t ELORDER >
const auto& QuadratureGenerator< QTYPE, QORDER >::get(const Element< ElementTypes::Line, ELORDER >&) const
{
    return ReferenceQuadrature< QTYPE, QORDER >::value;
}

template < QuadratureTypes QTYPE, q_o_t QORDER >
template < el_o_t ELORDER >
const auto& QuadratureGenerator< QTYPE, QORDER >::get(const Element< ElementTypes::Quad, ELORDER >&) const
{
    static const auto quad = [] {
        using ref_quadrature_t  = ReferenceQuadrature< QTYPE, QORDER >;
        constexpr auto ref_size = ref_quadrature_t::size;
        using quadrature_t      = Quadrature< ref_size * ref_size, 2 >;

        const auto& ref_quadrature   = ref_quadrature_t::value;
        const auto& ref_quadrature_p = ref_quadrature.getQPoints();
        const auto& ref_quadrature_w = ref_quadrature.getWeights();

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;

        size_t index = 0;
        for (size_t i = 0; i < ref_size; ++i)
        {
            for (size_t j = 0; j < ref_size; ++j)
            {
                q_points[index][0] = ref_quadrature_p[i][0];
                q_points[index][1] = ref_quadrature_p[j][0];
                weights[index]     = ref_quadrature_w[i] * ref_quadrature_w[j];
                ++index;
            }
        }

        return quadrature_t{q_points, weights};
    }();

    return quad;
}

template < QuadratureTypes QTYPE, q_o_t QORDER >
template < el_o_t ELORDER >
const auto& QuadratureGenerator< QTYPE, QORDER >::get(const Element< ElementTypes::Hex, ELORDER >&) const
{
    static const auto quad = [] {
        using ref_quadrature_t  = ReferenceQuadrature< QTYPE, QORDER >;
        constexpr auto ref_size = ref_quadrature_t::size;
        using quadrature_t      = Quadrature< ref_size * ref_size * ref_size, 3 >;

        const auto& ref_quadrature   = ref_quadrature_t::value;
        const auto& ref_quadrature_p = ref_quadrature.getQPoints();
        const auto& ref_quadrature_w = ref_quadrature.getWeights();

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;

        size_t index = 0;
        for (size_t i = 0; i < ref_size; ++i)
        {
            for (size_t j = 0; j < ref_size; ++j)
            {
                for (size_t k = 0; k < ref_size; ++k)
                {
                    q_points[index][0] = ref_quadrature_p[i][0];
                    q_points[index][1] = ref_quadrature_p[j][0];
                    q_points[index][2] = ref_quadrature_p[k][0];
                    weights[index]     = ref_quadrature_w[i] * ref_quadrature_w[j] * ref_quadrature_w[k];
                    ++index;
                }
            }
        }

        return quadrature_t{q_points, weights};
    }();

    return quad;
}
} // namespace lstr
#endif // L3STER_QUAD_QUADRATUREGENERATOR_HPP
