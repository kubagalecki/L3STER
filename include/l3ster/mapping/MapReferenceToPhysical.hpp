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
/// Map point in reference space to physical space
template < mesh::ElementType T, el_o_t O >
auto mapToPhysicalSpace(const mesh::ElementData< T, O >&                  element_data,
                        const Point< mesh::Element< T, O >::native_dim >& point) -> Point< 3 >
    requires(mesh::isGeomType(T))
{
    constexpr auto GBT             = basis::BasisType::Lagrange;
    constexpr auto GT              = mesh::ElementTraits< mesh::Element< T, O > >::geom_type;
    constexpr auto GO              = mesh::ElementTraits< mesh::Element< T, O > >::geom_order;
    const auto     geom_basis_vals = basis::computeReferenceBases< GT, GO, GBT >(point);
    const auto     verts_mat       = element_data.getEigenMap();
    const auto     phys_coords     = (verts_mat * geom_basis_vals).eval();
    return Point{phys_coords[0], phys_coords[1], phys_coords[2]};
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
