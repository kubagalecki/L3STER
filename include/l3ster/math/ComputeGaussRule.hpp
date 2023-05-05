#ifndef L3STER_MATH_COMPUTEGAUSSRULE_HPP
#define L3STER_MATH_COMPUTEGAUSSRULE_HPP

#include "l3ster/defs/Typedefs.h"
#include "l3ster/util/Concepts.hpp"

#include "l3ster/util/EigenUtils.hpp"

#include <cmath>
#include <utility>

namespace lstr
{
/*
usage: const auto [qp, w] = computeGaussRule<N>(a, b, c)

Compute the N-point Gauss quadrature rule (abscissas qp and weights w) over the interval [-1, 1] for
an orthogonal polynomial family given by the recurrence relation:
p_{n}(x) = (a(n)*x + b(n))*p_{n - 1}(x) - c(n)*p_{n - 2}(x)

Reference:
Golub, G. H., & Welsch, J. H. (1969). Calculation of Gauss quadrature rules.
Mathematics of Computation, 23(106), 221–221. https://doi.org/10.1090/s0025-5718-69-99647-1
*/
template < size_t order >
auto computeGaussRule(Mapping_c< size_t, val_t > auto&& a,
                      Mapping_c< size_t, val_t > auto&& b,
                      Mapping_c< size_t, val_t > auto&& c,
                      std::integral_constant< size_t, order > = {})
{
    static_assert(order > 0u);
    using matrix_t = Eigen::Matrix< val_t, order, order >;

    // Note: variable names follow the reference
    matrix_t J = matrix_t::Zero();

    for (size_t ind = 0; ind < order - 1; ++ind)
    {
        const auto  n     = ind + 1;
        const val_t alpha = -b(n) / a(n);
        const val_t beta  = sqrt(c(n + 1u) / (a(n) * a(n + 1u)));
        J(ind, ind)       = alpha;
        J(ind + 1, ind)   = beta;
        J(ind, ind + 1)   = beta;
    }
    J(order - 1, order - 1) = -b(order) / a(order);

    auto eigen_solver = Eigen::SelfAdjointEigenSolver< matrix_t >{};
    eigen_solver.compute(J);
    const auto& eig_vals = eigen_solver.eigenvalues();
    const auto& eig_vecs = eigen_solver.eigenvectors();

    std::array< val_t, order > points;
    std::array< val_t, order > weights;

    for (size_t i = 0; i < order; ++i)
    {
        points[i]  = eig_vals[i];
        weights[i] = 2. * eig_vecs(0, i) * eig_vecs(0, i);
    }

    return std::make_pair(points, weights);
}
} // namespace lstr

#endif // L3STER_MATH_COMPUTEGAUSSRULE_HPP
