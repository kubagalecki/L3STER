#ifndef L3STER_BASIS_TABULATEDBASIS_HPP
#define L3STER_BASIS_TABULATEDBASIS_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"

namespace lstr::basis
{
template < size_t num_bases, size_t num_points >
struct TabulatedBasisValues
{
private:
    using map_t       = Eigen::Map< Eigen::Matrix< val_t, num_points, num_bases > >;
    using const_map_t = Eigen::Map< const Eigen::Matrix< val_t, num_points, num_bases > >;

public:
    auto getMap() const noexcept { return const_map_t{values.data()}; }
    auto getMap() noexcept { return map_t{values.data()}; }

    alignas(64) std::array< val_t, num_points * num_bases > values;
};

template < size_t num_bases, size_t num_points, dim_t dim >
struct TabulatedBasisDerivatives
{
private:
    static constexpr auto getStride()
    {
        constexpr auto cache_size      = 64uz / sizeof(val_t);
        constexpr auto stride_unpadded = num_bases * num_points;
        constexpr auto rem64           = stride_unpadded % cache_size;
        if constexpr (rem64 == 0)
            return stride_unpadded;
        else
            return stride_unpadded - rem64 + cache_size;
    }
    static constexpr auto stride = getStride();

    using map_t       = Eigen::Map< Eigen::Matrix< val_t, num_points, num_bases > >;
    using const_map_t = Eigen::Map< const Eigen::Matrix< val_t, num_points, num_bases > >;

    // Stride for all derivatives at given point
    using ders_stride_t = Eigen::Stride< stride, num_points >;
    using ders_map_t    = Eigen::Map< const Eigen::Matrix< val_t, num_bases, dim >, Eigen::Unaligned, ders_stride_t >;

public:
    static_assert(num_points > 0 and dim > 0 and dim <= 3);

    auto getMap(dim_t dim_ind) const noexcept { return const_map_t{std::next(derivatives.data(), dim_ind * stride)}; }
    auto getMap(dim_t dim_ind) noexcept { return map_t{std::next(derivatives.data(), dim_ind * stride)}; }

    auto getPointMap(size_t point) const noexcept { return ders_map_t{std::next(derivatives.data(), point)}; }

    alignas(64) std::array< val_t, dim * stride > derivatives;
};

template < size_t num_bases, size_t num_points, dim_t dim >
struct TabulatedBasis
{
    static_assert(num_points > 0 and dim > 0 and dim <= 3);

    static constexpr size_t bases     = num_bases;
    static constexpr size_t size      = num_points;
    static constexpr dim_t  dimension = dim;

    auto getValuesMap() const noexcept { return values.getMap(); }
    auto getValuesMap() noexcept { return values.getMap(); }
    auto getDerivativesMap(dim_t dim_ind) const noexcept { return derivatives.getMap(dim_ind); }
    auto getDerivativesMap(dim_t dim_ind) noexcept { return derivatives.getMap(dim_ind); }

    TabulatedBasisValues< num_bases, num_points >           values;
    TabulatedBasisDerivatives< num_bases, num_points, dim > derivatives;
};

template < BasisType BT, mesh::ElementType ET, el_o_t EO, size_t NP >
auto tabulateBasis(std::span< const Point< mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim >, NP > points)
    requires(NP != std::dynamic_extent)
{
    using eltraits           = mesh::ElementTraits< mesh::Element< ET, EO > >;
    constexpr auto nat_dim   = eltraits::native_dim;
    constexpr auto num_bases = eltraits::nodes_per_element;
    constexpr auto GT        = eltraits::geom_type;

    auto retval = TabulatedBasis< num_bases, NP, nat_dim >{};
    auto vals   = retval.getValuesMap();
    for (auto&& [i, qp] : points | std::views::enumerate)
    {
        const auto qv = computeReferenceBases< GT, EO, BT >(qp);
        const auto qd = computeReferenceBasisDerivatives< GT, EO, BT >(qp);
        vals.row(i)   = qv.transpose();
        for (dim_t dim = 0; dim != nat_dim; ++dim)
            retval.getDerivativesMap(dim).row(i) = qd.row(dim);
    }
    return retval;
}
} // namespace lstr::basis
#endif // L3STER_BASIS_TABULATEDBASIS_HPP
