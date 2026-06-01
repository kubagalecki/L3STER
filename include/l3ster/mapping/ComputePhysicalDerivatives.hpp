#ifndef L3STER_MAPPING_COMPUTEPHYSICALDERIVATIVES_HPP
#define L3STER_MAPPING_COMPUTEPHYSICALDERIVATIVES_HPP

#include "l3ster/basisfun/BasisAtPoints.hpp"

namespace lstr::map
{
// Note: the following attempts to vectorize across (quadrature) points
// The compiler is able to autovectorize the loops without difficulty
template < size_t NP, dim_t D >
struct JacobiMats
{
    auto get(size_t point_ind) const noexcept -> Eigen::Matrix< val_t, D, D >
    {
        auto retval = Eigen::Matrix< val_t, D, D >{};
        for (dim_t c = 0; c != D; ++c)
            for (dim_t r = 0; r != D; ++r)
                retval(r, c) = elems[r + c * D][point_ind];
        return retval;
    }

    std::array< std::array< val_t, NP >, D * D > elems; // SoA of matrix elements
};

template < size_t NG, size_t NP, dim_t D >
auto computeJacobiMats(const basis::TabulatedBasisDerivatives< NG, NP, D >& geom_basis_ders,
                       std::span< const Point< 3 >, NG >                    vertices) -> JacobiMats< NP, D >
{
    JacobiMats< NP, D > retval;
    { // Assign on first loop iteration
        constexpr size_t       b = 0;
        std::array< val_t, D > vert;
        for (size_t i = 0; i != D; ++i)
            vert[i] = vertices[b][i];
        for (size_t p = 0; p != NP; ++p)
            for (dim_t r = 0; r != D; ++r)
                for (dim_t c = 0; c != D; ++c)
                    retval.elems[(r + c * D)][p] = vert[c] * geom_basis_ders.getMap(r)(p, b);
    }
    for (size_t b = 1; b != NG; ++b) // Now accumulate
    {
        std::array< val_t, D > vert;
        for (size_t i = 0; i != D; ++i)
            vert[i] = vertices[b][i];
        for (size_t p = 0; p != NP; ++p)
            for (dim_t r = 0; r != D; ++r)
                for (dim_t c = 0; c != D; ++c)
                    retval.elems[(r + c * D)][p] += vert[c] * geom_basis_ders.getMap(r)(p, b);
    }
    return retval;
}

template < size_t NP, dim_t D >
auto computeJacobians(const JacobiMats< NP, D >& jacobi_mats) -> std::array< val_t, NP >
{
    std::array< val_t, NP > retval;
    if constexpr (D == 1)
        std::copy_n(jacobi_mats.elems.front().data(), NP, retval.data());
    else if constexpr (D == 2)
    {
        const auto& [j00, j10, j01, j11] = jacobi_mats.elems;
        for (size_t i = 0; i != NP; ++i)
            retval[i] = j00[i] * j11[i] - j01[i] * j10[i];
    }
    else if constexpr (D == 3)
    {
        const auto& [j00, j10, j20, j01, j11, j21, j02, j12, j22] = jacobi_mats.elems;
        for (size_t i = 0; i != NP; ++i)
            retval[i] = j00[i] * (j11[i] * j22[i] - j12[i] * j21[i]) - j01[i] * (j10[i] * j22[i] - j12[i] * j20[i]) +
                        j02[i] * (j10[i] * j21[i] - j11[i] * j20[i]);
    }
    else
        static_assert(util::always_false< D >);
    return retval;
}

template < size_t NP, dim_t D >
void invertJacobiMats(JacobiMats< NP, D >& jacobi_mats, std::span< const val_t, NP > jacobians)
{
    if constexpr (D == 1)
        std::ranges::transform(
            jacobi_mats.elems.front(), jacobi_mats.elems.front().begin(), std::bind_front(std::divides{}, 1.));
    else if constexpr (D == 2)
    {
        auto& [j00, j10, j01, j11] = jacobi_mats.elems;
        for (size_t i = 0; i != NP; ++i)
        {
            const auto det_inv = 1. / jacobians[i];
            const auto j00_    = j00[i];
            const auto j01_    = j01[i];
            const auto j10_    = j10[i];
            const auto j11_    = j11[i];
            j00[i]             = det_inv * j11_;
            j10[i]             = -det_inv * j10_;
            j01[i]             = -det_inv * j01_;
            j11[i]             = det_inv * j00_;
        }
    }
    else if constexpr (D == 3)
    {
        auto& [j00, j10, j20, j01, j11, j21, j02, j12, j22] = jacobi_mats.elems;
        for (size_t i = 0; i != NP; ++i)
        {
            const auto det_inv = 1. / jacobians[i];
            const auto j00_    = j00[i];
            const auto j01_    = j01[i];
            const auto j02_    = j02[i];
            const auto j10_    = j10[i];
            const auto j11_    = j11[i];
            const auto j12_    = j12[i];
            const auto j20_    = j20[i];
            const auto j21_    = j21[i];
            const auto j22_    = j22[i];
            j00[i]             = det_inv * (j11_ * j22_ - j12_ * j21_);
            j10[i]             = -det_inv * (j10_ * j22_ - j12_ * j20_);
            j20[i]             = det_inv * (j10_ * j21_ - j11_ * j20_);
            j01[i]             = -det_inv * (j01_ * j22_ - j02_ * j21_);
            j11[i]             = det_inv * (j00_ * j22_ - j02_ * j20_);
            j21[i]             = -det_inv * (j00_ * j21_ - j01_ * j20_);
            j02[i]             = det_inv * (j01_ * j12_ - j02_ * j11_);
            j12[i]             = -det_inv * (j00_ * j12_ - j02_ * j10_);
            j22[i]             = det_inv * (j00_ * j11_ - j01_ * j10_);
        }
    }
    else
        static_assert(util::always_false< D >);
}

template < size_t NA, size_t NP, dim_t D >
auto computePhysicalDerivatives(const basis::TabulatedBasisDerivatives< NA, NP, D >& basis,
                                const JacobiMats< NP, D >&                           inverted_jacobi_mats)
    -> basis::TabulatedBasisDerivatives< NA, NP, D >
{
    basis::TabulatedBasisDerivatives< NA, NP, D > retval;
    for (size_t b = 0; b < NA; ++b)
        for (size_t p = 0; p != NP; ++p)
            for (dim_t r = 0; r != D; ++r)
            {
                val_t der = inverted_jacobi_mats.elems[r][p] * basis.getMap(0)(p, b);
                for (dim_t c = 1; c != D; ++c)
                    der += inverted_jacobi_mats.elems[(r + c * D)][p] * basis.getMap(c)(p, b);
                retval.getMap(r)(p, b) = der;
            }
    return retval;
}
} // namespace lstr::map
#endif // L3STER_MAPPING_COMPUTEPHYSICALDERIVATIVES_HPP
