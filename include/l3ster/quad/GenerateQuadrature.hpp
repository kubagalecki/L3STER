// Quadrature generation functionality

#ifndef L3STER_QUAD_GENERATEQUADRATURE_HPP
#define L3STER_QUAD_GENERATEQUADRATURE_HPP

#include "ReferenceQuadrature.hpp"
#include "l3ster/mesh/Element.hpp"

namespace lstr
{
template < QuadratureTypes QT, q_o_t QO, ElementTypes ET >
const auto& getQuadrature() requires(ET == ElementTypes::Line)
{
    return getReferenceQuadrature< QT, QO >();
}

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET >
const auto& getQuadrature() requires(ET == ElementTypes::Quad)
{
    static const auto value = [] {
        const auto& ref_quadrature   = getReferenceQuadrature< QT, QO >();
        const auto& ref_quadrature_p = ref_quadrature.getPoints();
        const auto& ref_quadrature_w = ref_quadrature.getWeights();

        constexpr auto ref_size = ref_quadrature.size;
        using quadrature_t      = Quadrature< ref_size * ref_size, 2 >;

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;

        size_t index = 0;
        for (size_t i = 0; i < ref_size; ++i)
            for (size_t j = 0; j < ref_size; ++j)
            {
                q_points[index][0] = ref_quadrature_p[i][0];
                q_points[index][1] = ref_quadrature_p[j][0];
                weights[index]     = ref_quadrature_w[i] * ref_quadrature_w[j];
                ++index;
            }

        return quadrature_t{q_points, weights};
    }();
    return value;
}

template < QuadratureTypes QT, q_o_t QO, ElementTypes ET >
const auto& getQuadrature() requires(ET == ElementTypes::Hex)
{
    static const auto value = [] {
        const auto& ref_quadrature   = getReferenceQuadrature< QT, QO >();
        const auto& ref_quadrature_p = ref_quadrature.getPoints();
        const auto& ref_quadrature_w = ref_quadrature.getWeights();

        constexpr auto ref_size = ref_quadrature.size;
        using quadrature_t      = Quadrature< ref_size * ref_size * ref_size, 3 >;

        typename quadrature_t::q_points_t q_points;
        typename quadrature_t::weights_t  weights;

        size_t index = 0;
        for (size_t i = 0; i < ref_size; ++i)
            for (size_t j = 0; j < ref_size; ++j)
                for (size_t k = 0; k < ref_size; ++k)
                {
                    q_points[index][0] = ref_quadrature_p[i][0];
                    q_points[index][1] = ref_quadrature_p[j][0];
                    q_points[index][2] = ref_quadrature_p[k][0];
                    weights[index]     = ref_quadrature_w[i] * ref_quadrature_w[j] * ref_quadrature_w[k];
                    ++index;
                }

        return quadrature_t{q_points, weights};
    }();
    return value;
}
} // namespace lstr
#endif // L3STER_QUAD_GENERATEQUADRATURE_HPP
