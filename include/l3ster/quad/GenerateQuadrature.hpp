// Quadrature generation functionality

#ifndef L3STER_QUAD_GENERATEQUADRATURE_HPP
#define L3STER_QUAD_GENERATEQUADRATURE_HPP

#include "l3ster/mesh/Element.hpp"
#include "l3ster/quad/ReferenceQuadrature.hpp"

namespace lstr::quad
{
template < QuadratureType QT, q_o_t QO, mesh::ElementType ET >
const auto& getQuadrature()
    requires(ET == mesh::ElementType::Line)
{
    return getReferenceQuadrature< QT, QO >();
}

template < QuadratureType QT, q_o_t QO, mesh::ElementType ET >
const auto& getQuadrature()
    requires(ET == mesh::ElementType::Quad)
{
    static const auto value = std::invoke([] {
        const auto& ref_quadrature   = getReferenceQuadrature< QT, QO >();
        const auto& ref_quadrature_p = ref_quadrature.points;
        const auto& ref_quadrature_w = ref_quadrature.weights;

        constexpr auto ref_size = ref_quadrature.size;
        using quadrature_t      = Quadrature< ref_size * ref_size, 2 >;

        auto q_points = typename quadrature_t::q_points_t{};
        auto weights  = typename quadrature_t::weights_t{};

        constexpr auto iter1 = std::views::iota(0uz, ref_size);
        for (auto&& [index, ij] : std::views::cartesian_product(iter1, iter1) | std::views::enumerate)
        {
            const auto [i, j]  = ij;
            const auto induz   = static_cast< size_t >(index);
            q_points[induz][0] = ref_quadrature_p[i][0];
            q_points[induz][1] = ref_quadrature_p[j][0];
            weights[induz]     = ref_quadrature_w[i] * ref_quadrature_w[j];
        }
        return quadrature_t{q_points, weights};
    });
    return value;
}

template < QuadratureType QT, q_o_t QO, mesh::ElementType ET >
const auto& getQuadrature()
    requires(ET == mesh::ElementType::Hex)
{
    static const auto value = std::invoke([] {
        const auto& ref_quadrature   = getReferenceQuadrature< QT, QO >();
        const auto& ref_quadrature_p = ref_quadrature.points;
        const auto& ref_quadrature_w = ref_quadrature.weights;

        constexpr auto ref_size = ref_quadrature.size;
        using quadrature_t      = Quadrature< ref_size * ref_size * ref_size, 3 >;

        auto q_points = typename quadrature_t::q_points_t{};
        auto weights  = typename quadrature_t::weights_t{};

        constexpr auto iter1 = std::views::iota(0uz, ref_size);
        for (auto&& [index, ijk] : std::views::cartesian_product(iter1, iter1, iter1) | std::views::enumerate)
        {
            const auto [i, j, k] = ijk;
            const auto induz     = static_cast< size_t >(index);
            q_points[induz][0]   = ref_quadrature_p[i][0];
            q_points[induz][1]   = ref_quadrature_p[j][0];
            q_points[induz][2]   = ref_quadrature_p[k][0];
            weights[induz]       = ref_quadrature_w[i] * ref_quadrature_w[j] * ref_quadrature_w[k];
        }
        return quadrature_t{q_points, weights};
    });
    return value;
}
} // namespace lstr::quad
#endif // L3STER_QUAD_GENERATEQUADRATURE_HPP
