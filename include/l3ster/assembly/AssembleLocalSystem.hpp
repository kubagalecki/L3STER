#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDersAtPoints.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"

namespace lstr
{
namespace detail
{
template < typename T >
concept ValidKernelResult_c = Pair_c< T > and Array_c< typename T::first_type > and
                              EigenMatrix_c< typename T::first_type::value_type > and
                              EigenVector_c< typename T::second_type > and
                              (T::first_type::value_type::RowsAtCompileTime == T::second_type::RowsAtCompileTime);

template < typename K, dim_t dim, size_t n_fields >
concept Kernel_c = requires(K                                                kernel,
                            std::array< val_t, n_fields >                    node_vals,
                            std::array< std::array< val_t, n_fields >, dim > node_ders,
                            SpaceTimePoint                                   point) {
                       {
                           std::invoke(kernel, node_vals, node_ders, point)
                           } noexcept -> ValidKernelResult_c;
                   };

template < typename K, dim_t dim, size_t n_fields >
concept BoundaryKernel_c = requires(K                                                kernel,
                                    std::array< val_t, n_fields >                    node_vals,
                                    std::array< std::array< val_t, n_fields >, dim > node_ders,
                                    SpaceTimePoint                                   point,
                                    Eigen::Vector< val_t, dim >                      normal) {
                               {
                                   std::invoke(kernel, node_vals, node_ders, point, normal)
                                   } noexcept -> ValidKernelResult_c;
                           };

template < typename Kernel, dim_t dim, size_t n_fields >
    requires Kernel_c< Kernel, dim, n_fields > or BoundaryKernel_c< Kernel, dim, n_fields >
struct kernel_result
{};
template < typename Kernel, dim_t dim, size_t n_fields >
    requires Kernel_c< Kernel, dim, n_fields >
struct kernel_result< Kernel, dim, n_fields >
{
    using type = std::invoke_result_t< Kernel,
                                       std::array< val_t, n_fields >,
                                       std::array< std::array< val_t, n_fields >, dim >,
                                       SpaceTimePoint >;
};
template < typename Kernel, dim_t dim, size_t n_fields >
    requires BoundaryKernel_c< Kernel, dim, n_fields >
struct kernel_result< Kernel, dim, n_fields >
{
    using type = std::invoke_result_t< Kernel,
                                       std::array< val_t, n_fields >,
                                       std::array< std::array< val_t, n_fields >, dim >,
                                       SpaceTimePoint,
                                       Eigen::Vector< val_t, dim > >;
};
template < typename Kernel, dim_t dim, size_t n_fields >
    requires Kernel_c< Kernel, dim, n_fields > or BoundaryKernel_c< Kernel, dim, n_fields >
using kernel_result_t = typename kernel_result< Kernel, dim, n_fields >::type;
template < typename Kernel, dim_t dim, size_t n_fields >
inline constexpr std::size_t n_unknowns =
    kernel_result_t< Kernel, dim, n_fields >::first_type::value_type::ColsAtCompileTime;

template < int n_nodes, int n_fields, size_t n_qp, int dim >
auto computeFieldValsAndDers(const std::array< Eigen::Vector< val_t, n_nodes >, n_qp >&            basis_vals,
                             const std::array< EigenRowMajorMatrix< val_t, dim, n_nodes >, n_qp >& basis_ders,
                             const EigenRowMajorMatrix< val_t, n_nodes, n_fields >&                node_vals)
{
    using quantity_at_qps = std::array< std::array< val_t, n_fields >, n_qp >;
    std::pair< quantity_at_qps, std::array< quantity_at_qps, dim > > retval;
    auto& [field_vals, field_ders] = retval;
    if constexpr (n_fields != 0)
    {
        for (size_t qp_ind = 0; qp_ind < n_qp; ++qp_ind)
        {
            const Eigen::Vector< val_t, n_fields > field_vals_packed = node_vals.transpose() * basis_vals[qp_ind];
            const EigenRowMajorMatrix< val_t, dim, n_fields > field_ders_packed = basis_ders[qp_ind] * node_vals;
            std::ranges::copy(field_vals_packed, begin(field_vals[qp_ind]));
            for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
                std::copy_n(
                    field_ders_packed.data() + dim_ind * n_fields, n_fields, field_ders[dim_ind][qp_ind].begin());
        }
    }
    return retval;
}

template < size_t n_fields, size_t n_qp, size_t dim >
auto extractFieldValsAndDersAtQpoint(
    const std::pair< std::array< std::array< val_t, n_fields >, n_qp >,
                     std::array< std::array< std::array< val_t, n_fields >, n_qp >, dim > >& field_vals_and_ders,
    size_t                                                                                   qp_ind)
{
    using quantity_at_qp = std::array< val_t, n_fields >;
    std::pair< quantity_at_qp, std::array< quantity_at_qp, dim > > retval;

    const auto& [field_vals_all, field_ders_all] = field_vals_and_ders;
    auto& [field_vals, field_ders]               = retval;
    field_vals                                   = field_vals_all[qp_ind];
    for (size_t dim_ind = 0; dim_ind < dim; ++dim_ind)
        field_ders[dim_ind] = field_ders_all[dim_ind][qp_ind];
    return retval;
}

SpaceTimePoint makeSpaceTimePoint(const auto& element, const auto& quadrature, ptrdiff_t qp_ind, val_t time)
{
    return SpaceTimePoint{.space = mapToPhysicalSpace(element, Point{quadrature.points[qp_ind]}), .time = time};
}

template < typename Kernel >
auto evaluateKernel(Kernel&&    kernel,
                    const auto& element,
                    const auto& field_vals_and_ders,
                    const auto& quadrature,
                    ptrdiff_t   qp_ind,
                    val_t       time)
{
    const auto [field_vals_at_qp, field_ders_at_qp] = extractFieldValsAndDersAtQpoint(field_vals_and_ders, qp_ind);
    const auto space_time                           = makeSpaceTimePoint(element, quadrature, qp_ind, time);
    return std::invoke(std::forward< Kernel >(kernel), field_vals_at_qp, field_ders_at_qp, space_time);
}

template < typename Kernel >
auto evaluateBoundaryKernel(Kernel&&    kernel,
                            const auto& el_view,
                            const auto& field_vals_and_ders,
                            const auto& quadrature,
                            ptrdiff_t   qp_ind,
                            val_t       time,
                            const auto& normal)
{
    const auto [field_vals_at_qp, field_ders_at_qp] = extractFieldValsAndDersAtQpoint(field_vals_and_ders, qp_ind);
    const auto space_time                           = makeSpaceTimePoint(*el_view, quadrature, qp_ind, time);
    return std::invoke(std::forward< Kernel >(kernel), field_vals_at_qp, field_ders_at_qp, space_time, normal);
}

auto makeRankUpdateMatrix(const auto& kernel_result, const auto& basis_vals, const auto& basis_ders, ptrdiff_t qp_ind)
{
    constexpr size_t n_equations = std::remove_cvref_t< decltype(kernel_result) >::value_type::RowsAtCompileTime;
    constexpr size_t n_unknowns  = std::remove_cvref_t< decltype(kernel_result) >::value_type::ColsAtCompileTime;
    constexpr size_t n_bases     = std::remove_cvref_t< decltype(basis_vals) >::value_type::RowsAtCompileTime;
    constexpr size_t dim         = std::remove_cvref_t< decltype(basis_ders) >::value_type::RowsAtCompileTime;
    EigenRowMajorMatrix< val_t, n_bases * n_unknowns, n_equations > retval;
    for (size_t basis_ind = 0; basis_ind < n_bases; ++basis_ind)
    {
        retval(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) =
            basis_vals[qp_ind][basis_ind] * kernel_result[0].transpose();
        for (size_t dim_ind = 0; dim_ind < dim; ++dim_ind)
            retval(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) +=
                basis_ders[qp_ind](dim_ind, basis_ind) * kernel_result[dim_ind + 1].transpose();
    }
    return retval;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
auto initLocalSystem()
{
    constexpr auto dim                = Element< ET, EO >::native_dim;
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns< Kernel, dim, n_fields >;
    using k_el_t                      = EigenRowMajorMatrix< val_t, local_problem_size, local_problem_size >;
    using f_el_t                      = Eigen::Vector< val_t, local_problem_size >;
    return std::pair< k_el_t, f_el_t >{k_el_t::Zero(), f_el_t::Zero()};
}
} // namespace detail

/*
template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalSystem2(Kernel&&                                                                       kernel,
                          const Element< ET, EO >&                                                       element,
                          const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >&      node_vals,
                          const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_q,
                          val_t                                                                          time)
    requires detail::Kernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto& quadrature           = basis_at_q.quadrature;
    const auto& basis_vals           = basis_at_q.basis.values;
    const auto  jacobi_mat_generator = getNatJacobiMatGenerator(element);

    auto local_system  = detail::initLocalSystem< Kernel, ET, EO, n_fields >();
    auto& [K_el, F_el] = local_system;
    for (size_t qp_ind = 0; qp_ind < quadrature.size; ++qp_ind)
    {
        const auto quad_point           = quadrature.points[qp_ind];
        const auto quad_weight          = quadrature.weights[qp_ind];
        const auto basis_vals_at_qp     = basis_vals[qp_ind];
        const auto ref_basis_ders_at_qp = basis_ders[qp_ind];
        const auto jacobi_mat           = jacobi_mat_generator(quad_point);
        const auto phys_basis_ders      = computePhysBasisDers(ref_basis_ders_at_qp, jacobi_mat);
        const auto fields_at_qp         = detail::computeFieldValsAndDers(node_vals, basis_vals_at_qp, phys_basis_ders);
        const auto physical_coords      = mapToPhysicalSpace(element, quad_point);
        const auto [A, F]               = detail::evaluateKernel(kernel, fields_at_qp, physical_coords, time);
        const auto rank_update_matrix   = detail::makeRankUpdateMatrix(A, basis_vals, phys_basis_ders);
        const auto rank_update_weight   = jacobi_mat.determinant() * quad_weight;
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
    }

    K_el = K_el.template selfadjointView< Eigen::Lower >();
    return local_system;
}
 */

template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalSystem(Kernel&&                                                                       kernel,
                         const Element< ET, EO >&                                                       element,
                         const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >&      node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_q,
                         val_t                                                                          time)
    requires detail::Kernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto& quadrature = basis_at_q.quadrature;
    const auto& basis_vals = basis_at_q.basis.values;

    const auto jac_at_qp           = computeJacobiansAtPoints(element, quadrature.points);
    const auto basis_ders          = computePhysBasisDersAtPoints(basis_at_q.basis.derivatives, jac_at_qp);
    const auto field_vals_and_ders = detail::computeFieldValsAndDers(basis_vals, basis_ders, node_vals);

    auto local_system     = detail::initLocalSystem< Kernel, ET, EO, n_fields >();
    auto& [K_el, F_el]    = local_system;
    const auto process_qp = [&](ptrdiff_t qp_ind) {
        const auto [A, F] = detail::evaluateKernel(kernel, element, field_vals_and_ders, quadrature, qp_ind, time);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, basis_vals, basis_ders, qp_ind);
        const auto rank_update_weight = jac_at_qp[qp_ind].determinant() * quadrature.weights[qp_ind];
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
    };

    for (ptrdiff_t qp_ind = 0; qp_ind < static_cast< ptrdiff_t >(quadrature.size); ++qp_ind)
        process_qp(qp_ind);
    K_el = K_el.template selfadjointView< Eigen::Lower >();
    return local_system;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalBoundarySystem(
    Kernel&&                                                                       kernel,
    BoundaryElementView< ET, EO >                                                  el_view,
    const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >&      node_vals,
    const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_q,
    val_t                                                                          time)
    requires detail::BoundaryKernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto& quadrature = basis_at_q.quadrature;
    const auto& basis_vals = basis_at_q.basis.values;

    const auto jac_at_qp           = computeJacobiansAtPoints(*el_view, quadrature.points);
    const auto basis_ders          = computePhysBasisDersAtPoints(basis_at_q.basis.derivatives, jac_at_qp);
    const auto field_vals_and_ders = detail::computeFieldValsAndDers(basis_vals, basis_ders, node_vals);

    auto local_system     = detail::initLocalSystem< Kernel, ET, EO, n_fields >();
    auto& [K_el, F_el]    = local_system;
    const auto process_qp = [&](ptrdiff_t qp_ind) {
        const auto normal = computeBoundaryNormal(el_view, jac_at_qp[qp_ind]);
        const auto [A, F] =
            detail::evaluateBoundaryKernel(kernel, el_view, field_vals_and_ders, quadrature, qp_ind, time, normal);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, basis_vals, basis_ders, qp_ind);
        const auto rank_update_weight = jac_at_qp[qp_ind].determinant() * quadrature.weights[qp_ind];
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
    };

    for (ptrdiff_t qp_ind = 0; qp_ind < static_cast< ptrdiff_t >(quadrature.size); ++qp_ind)
        process_qp(qp_ind);
    K_el = K_el.template selfadjointView< Eigen::Lower >();
    return local_system;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
