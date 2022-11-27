#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/BoundaryIntegralJacobian.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/JacobiMat.hpp"
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

template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
auto initLocalSystem()
{
    constexpr auto dim                = Element< ET, EO >::native_dim;
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns< Kernel, dim, n_fields >;
    using k_el_t                      = EigenRowMajorMatrix< val_t, local_problem_size, local_problem_size >;
    using f_el_t                      = Eigen::Vector< val_t, local_problem_size >;
    using retval_payload_t            = std::pair< k_el_t, f_el_t >;

    auto retval    = std::make_unique< retval_payload_t >();
    retval->first  = k_el_t::Zero();
    retval->second = f_el_t::Zero();
    return retval;
}

template < int n_nodes, int n_fields >
auto computeFieldVals(const Eigen::Vector< val_t, n_nodes >&                 basis_vals,
                      const EigenRowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< val_t, n_fields > retval;
    if constexpr (n_fields != 0)
    {
        const Eigen::Vector< val_t, n_fields > field_vals_packed = node_vals.transpose() * basis_vals;
        std::ranges::copy(field_vals_packed, begin(retval));
    }
    return retval;
}

template < int n_nodes, int n_fields, int dim >
auto computeFieldDers(const EigenRowMajorMatrix< val_t, dim, n_nodes >&      basis_ders,
                      const EigenRowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< std::array< val_t, n_fields >, dim > retval;
    if constexpr (n_fields != 0)
    {
        const EigenRowMajorMatrix< val_t, dim, n_fields > field_ders_packed = basis_ders * node_vals;
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
            std::copy_n(std::next(field_ders_packed.data(), dim_ind * n_fields), n_fields, retval[dim_ind].begin());
    }
    return retval;
}

template < int n_equations, int n_unknowns, int n_bases, int dim >
auto makeRankUpdateMatrix(const std::array< Eigen::Matrix< val_t, n_equations, n_unknowns >, dim + 1 >& kernel_result,
                          const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                          const EigenRowMajorMatrix< val_t, dim, n_bases >&                             basis_ders)
{
    EigenRowMajorMatrix< val_t, n_bases * n_unknowns, n_equations > retval;
    for (size_t basis_ind = 0; basis_ind < static_cast< size_t >(n_bases); ++basis_ind)
    {
        retval(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) =
            basis_vals[basis_ind] * kernel_result[0].transpose();
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
            retval(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) +=
                basis_ders(dim_ind, basis_ind) * kernel_result[dim_ind + 1].transpose();
    }
    return retval;
}
} // namespace detail

template < ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalSystem(auto&&                                                                    kernel,
                         const Element< ET, EO >&                                                  element,
                         const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL >&                           basis_at_qps,
                         val_t                                                                     time)
    requires detail::Kernel_c< decltype(kernel), Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(element);
    auto       local_system         = detail::initLocalSystem< decltype(kernel), ET, EO, n_fields >();
    auto& [K_el, F_el]              = *local_system;

    const auto process_qp = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat         = jacobi_mat_generator(point);
        const auto phys_basis_ders    = computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals         = detail::computeFieldVals(bas_vals, node_vals);
        const auto field_ders         = detail::computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords        = mapToPhysicalSpace(element, point);
        const auto [A, F]             = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time});
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, bas_vals, phys_basis_ders);
        const auto rank_update_weight = jacobi_mat.determinant() * weight;
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
    };
    for (size_t qp_ind = 0; qp_ind < basis_at_qps.quadrature.size; ++qp_ind)
        process_qp(basis_at_qps.quadrature.points[qp_ind],
                   basis_at_qps.quadrature.weights[qp_ind],
                   basis_at_qps.basis.values[qp_ind],
                   basis_at_qps.basis.derivatives[qp_ind]);

    K_el = K_el.template selfadjointView< Eigen::Lower >();
    return local_system;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalBoundarySystem(Kernel&&                                                                  kernel,
                                 BoundaryElementView< ET, EO >                                             el_view,
                                 const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                                 const ReferenceBasisAtQuadrature< ET, EO, QL >&                           basis_at_qps,
                                 val_t                                                                     time)
    requires detail::BoundaryKernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(*el_view);
    auto       local_system         = detail::initLocalSystem< decltype(kernel), ET, EO, n_fields >();
    auto& [K_el, F_el]              = *local_system;

    const auto process_qp = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat      = jacobi_mat_generator(point);
        const auto phys_basis_ders = computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals      = detail::computeFieldVals(bas_vals, node_vals);
        const auto field_ders      = detail::computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords     = mapToPhysicalSpace(*el_view, point);
        const auto normal          = computeBoundaryNormal(el_view, jacobi_mat);
        const auto [A, F] = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time}, normal);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, bas_vals, phys_basis_ders);
        const auto bound_jac          = computeBoundaryIntegralJacobian(el_view, jacobi_mat);
        const auto rank_update_weight = bound_jac * weight;
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
    };
    for (size_t qp_ind = 0; qp_ind < basis_at_qps.quadrature.size; ++qp_ind)
        process_qp(basis_at_qps.quadrature.points[qp_ind],
                   basis_at_qps.quadrature.weights[qp_ind],
                   basis_at_qps.basis.values[qp_ind],
                   basis_at_qps.basis.derivatives[qp_ind]);

    K_el = K_el.template selfadjointView< Eigen::Lower >();
    return local_system;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
