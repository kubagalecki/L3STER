#ifndef L3STER_MAPPING_MAPREFERENCETOPHYSICAL_HPP
#define L3STER_MAPPING_MAPREFERENCETOPHYSICAL_HPP

#include "l3ster/basisfun/ReferenceBasisFunction.hpp"
#include "l3ster/mapping/BoundaryIntegralJacobian.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mesh/Element.hpp"
#include "l3ster/util/Caliper.hpp"

namespace lstr::map
{
namespace detail
{
template < basis::BasisType BT, mesh::ElementType T, el_o_t O, RandomAccessRangeOf< val_t > R >
val_t valueAt(R&& node_vals, const Point< mesh::Element< T, O >::native_dim >& point)
{
    const auto basis_vals = basis::computeReferenceBases< T, O, BT >(point);
    val_t      retval     = 0.;
    for (size_t i = 0; i != std::ranges::size(node_vals); ++i)
        retval += node_vals[i] * basis_vals[i];
    return retval;
}
} // namespace detail

/// Map point in reference space to physical space
template < mesh::ElementType T, el_o_t O >
auto mapToPhysicalSpace(const mesh::ElementData< T, O >&                  element_data,
                        const Point< mesh::Element< T, O >::native_dim >& point) -> Point< 3 >
    requires(util::contains({mesh::ElementType::Line, mesh::ElementType::Quad, mesh::ElementType::Hex}, T))
{
    const auto& vertices    = element_data.vertices;
    const auto  compute_dim = [&](ptrdiff_t dim) {
        return detail::valueAt< basis::BasisType::Lagrange, T, 1 >(
            vertices | std::views::transform([&](const Point< 3 >& p) { return p[dim]; }), point);
    };
    return Point{compute_dim(0), compute_dim(1), compute_dim(2)};
}

template < int n_bases, int dim >
struct DomainMappingResult
{
    template < typename JacobiMatGenerator >
    DomainMappingResult(JacobiMatGenerator&&                                      jacobi_gen,
                        const std::array< val_t, dim >&                           point,
                        const util::eigen::RowMajorMatrix< val_t, dim, n_bases >& ref_basis_ders)
    {
        const auto jacobi_mat = std::invoke(jacobi_gen, point);
        phys_basis_ders       = computePhysBasisDers(jacobi_mat, ref_basis_ders);
        jacobian              = jacobi_mat.determinant();
    }

    Eigen::Matrix< val_t, dim, n_bases > phys_basis_ders;
    val_t                                jacobian;
};

template < int n_bases, int dim >
struct BoundaryMappingResult
{
    template < mesh::ElementType ET, el_o_t EO, typename JacobiMatGenerator >
    BoundaryMappingResult(JacobiMatGenerator&&                                      jacobi_gen,
                          const std::array< val_t, dim >&                           point,
                          const util::eigen::RowMajorMatrix< val_t, dim, n_bases >& ref_basis_ders,
                          el_side_t                                                 side,
                          util::ValuePack< ET, EO >)
    {
        const auto jacobi_mat = std::invoke(jacobi_gen, point);
        phys_basis_ders       = computePhysBasisDers(jacobi_mat, ref_basis_ders);
        jacobian              = computeBoundaryIntegralJacobian< ET >(side, jacobi_mat);
        normal                = computeBoundaryNormal< ET, EO >(side, jacobi_mat);
    }

    Eigen::Matrix< val_t, dim, n_bases > phys_basis_ders;
    val_t                                jacobian;
    Eigen::Vector< val_t, dim >          normal;
};

template < mesh::ElementType ET, el_o_t EO >
using RefDersMat =
    util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::native_dim, mesh::Element< ET, EO >::n_nodes >;

template < mesh::ElementType ET, el_o_t EO, typename JacobiMatGenerator >
auto mapDomain(JacobiMatGenerator&&                                            jacobi_gen,
               const std::array< val_t, mesh::Element< ET, EO >::native_dim >& point,
               const RefDersMat< ET, EO >&                                     ref_basis_ders)
{
    L3STER_PROFILE_FUNCTION;
    using retval_t = DomainMappingResult< mesh::Element< ET, EO >::n_nodes, mesh::Element< ET, EO >::native_dim >;
    return retval_t{std::forward< JacobiMatGenerator >(jacobi_gen), point, ref_basis_ders};
}

template < mesh::ElementType ET, el_o_t EO, typename JacobiMatGenerator >
auto mapBoundary(JacobiMatGenerator&&                                            jacobi_gen,
                 const std::array< val_t, mesh::Element< ET, EO >::native_dim >& point,
                 const RefDersMat< ET, EO >&                                     ref_basis_ders,
                 el_side_t                                                       side)
{
    L3STER_PROFILE_FUNCTION;
    using retval_t = BoundaryMappingResult< mesh::Element< ET, EO >::n_nodes, mesh::Element< ET, EO >::native_dim >;
    constexpr auto el_typeinfo = util::ValuePack< ET, EO >{};
    return retval_t{std::forward< JacobiMatGenerator >(jacobi_gen), point, ref_basis_ders, side, el_typeinfo};
}
} // namespace lstr::map
#endif // L3STER_MAPPING_MAPREFERENCETOPHYSICAL_HPP
