#ifndef L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
#define L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP

#include <cmath>

namespace lstr::quad
{
template < QuadratureTypes QTYPE, types::q_o_t QORDER >
struct ReferenceQuadrature;

template < types::q_o_t QORDER >
struct ReferenceQuadrature< QuadratureTypes::GLeg, QORDER >
{
private:
    static auto compute();

public:
    using this_t       = ReferenceQuadrature< QuadratureTypes::GLeg, QORDER >;
    using quadrature_t = Quadrature< ReferenceQuadratureTraits< this_t >::size, 1 >;

    static inline const quadrature_t value = compute();
};

template < types::q_o_t QORDER >
auto ReferenceQuadrature< QuadratureTypes::GLeg, QORDER >::compute()
{
    constexpr auto size = ReferenceQuadratureTraits< this_t >::size;

    if constexpr (size == 1)
        return quadrature_t{{0.}, {2.}};

    using matrix_t = Eigen::Matrix< types::val_t, size, size >;

    matrix_t coef_matrix = matrix_t::Zero();

    for (size_t i = 0; i < size - 1; ++i)
    {
        const auto   i_inc1   = static_cast< types::val_t >(i + 1);
        types::val_t val      = i_inc1 / sqrt(4. * i_inc1 * i_inc1 - 1.);
        coef_matrix(i + 1, i) = val;
        coef_matrix(i, i + 1) = val;
    }

    Eigen::SelfAdjointEigenSolver< matrix_t > eigen_solver;
    eigen_solver.compute(coef_matrix);
    const auto& eig_vals = eigen_solver.eigenvalues();
    const auto& eig_vecs = eigen_solver.eigenvectors();

    typename quadrature_t::q_points_t q_points;
    typename quadrature_t::weights_t  weights;

    for (size_t i = 0; i < size; ++i)
    {
        q_points[i][0] = eig_vals[i];
        weights[i]     = 2. * eig_vecs(0, i) * eig_vecs(0, i);
    }

    return quadrature_t{q_points, weights};
}
} // namespace lstr::quad

#endif // L3STER_QUADRATURE_REFERENCEQUADRATURE_HPP
