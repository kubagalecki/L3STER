#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/local_assembly/SpaceTimePoint.hpp"
#include "l3ster/mapping/ComputePhysBasisDersAtQpoints.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"

namespace lstr
{
namespace detail
{
template < typename T >
concept ValidKernelResult_c = pair< T > and array< typename T::first_type > and
                              EigenMatrix_c< typename T::first_type::value_type > and
                              EigenMatrix_c< typename T::second_type > and (T::second_type::ColsAtCompileTime == 1) and
                              (T::first_type::value_type::RowsAtCompileTime == T::second_type::RowsAtCompileTime);

template < typename K, ElementTypes ET, el_o_t EO, size_t n_fields >
concept Kernel_c = requires(K                                                                          kernel,
                            std::array< val_t, n_fields >                                              node_vals,
                            std::array< std::array< val_t, n_fields >, Element< ET, EO >::native_dim > node_ders,
                            SpaceTimePoint                                                             point) {
                       {
                           std::invoke(kernel, node_vals, node_ders, point)
                           } noexcept -> ValidKernelResult_c;
                   };

template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
using kernel_return_t =
    std::invoke_result_t< Kernel,
                          std::array< val_t, n_fields >,
                          std::array< std::array< val_t, n_fields >, Element< ET, EO >::native_dim >,
                          SpaceTimePoint >;
template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
inline constexpr std::size_t n_unknowns =
    kernel_return_t< Kernel, ET, EO, n_fields >::first_type::value_type::ColsAtCompileTime;

template < int n_nodes, int n_fields, int n_qp, size_t n_dims >
auto computeFieldValsAndDers(const Eigen::Matrix< val_t, n_qp, n_nodes >&                       basis_vals,
                             const std::array< Eigen::Matrix< val_t, n_qp, n_nodes >, n_dims >& basis_ders,
                             const Eigen::Matrix< val_t, n_nodes, n_fields >&                   node_vals)
{
    using quantity_at_qps = Eigen::Matrix< val_t, n_qp, n_fields >;
    std::pair< quantity_at_qps, std::array< quantity_at_qps, n_dims > > retval;
    if constexpr (n_fields == 0)
        return retval;
    else
    {
        retval.first = basis_vals * node_vals;
        for (size_t dim = 0; dim < n_dims; ++dim)
            retval.second[dim] = basis_ders[dim] * node_vals;
        return retval;
    }
}

template < int n_fields, int n_qp, size_t n_dims >
auto extractFieldValsAndDersAtQpoint(
    const std::pair< Eigen::Matrix< val_t, n_qp, n_fields >,
                     std::array< Eigen::Matrix< val_t, n_qp, n_fields >, n_dims > >& field_vals_and_ders,
    size_t                                                                           qp_ind)
{
    using quantity_at_qp        = std::array< val_t, n_fields >;
    const auto extract_quantity = [&](const Eigen::Matrix< val_t, n_qp, n_fields >& quantity, quantity_at_qp& target) {
        for (size_t i = 0; i < n_fields; ++i)
            target[i] = quantity(qp_ind, i);
    };

    const auto& [field_vals, field_ders] = field_vals_and_ders;
    std::pair< quantity_at_qp, std::array< quantity_at_qp, n_dims > > retval;
    extract_quantity(field_vals, retval.first);
    for (size_t dim = 0; dim < n_dims; ++dim)
        extract_quantity(field_ders[dim], retval.second[dim]);
    return retval;
}

SpaceTimePoint makeSpaceTimePoint(const auto& element, const auto& quadrature, ptrdiff_t qp_ind, val_t time)
{
    return SpaceTimePoint{.space = mapToPhysicalSpace(element, Point{quadrature.getPoints()[qp_ind]}), .time = time};
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

auto makeRankUpdateMatrix(const auto& kernel_result, const auto& basis_vals, const auto& basis_ders, ptrdiff_t qp_ind)
{
    constexpr size_t n_equations = std::remove_cvref_t< decltype(kernel_result) >::value_type::RowsAtCompileTime;
    constexpr size_t n_unknowns  = std::remove_cvref_t< decltype(kernel_result) >::value_type::ColsAtCompileTime;
    constexpr size_t n_bases     = std::remove_cvref_t< decltype(basis_vals) >::ColsAtCompileTime;
    using ret_t                  = Eigen::Matrix< val_t, n_bases * n_unknowns, n_equations, Eigen::RowMajor >;
    ret_t ret_val; // NOLINT filled in directly below
    for (ptrdiff_t basis_ind = 0; basis_ind < static_cast< ptrdiff_t >(n_bases); ++basis_ind)
    {
        ret_val(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) =
            basis_vals(qp_ind, basis_ind) * kernel_result[0].transpose();
        for (ptrdiff_t dim = 1; const auto& basis_der : basis_ders)
            ret_val(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) +=
                basis_der(qp_ind, basis_ind) * kernel_result[dim++].transpose();
    }
    return ret_val;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
auto initLocalSystem()
{
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns< Kernel, ET, EO, n_fields >;
    using k_el_t                      = Eigen::Matrix< val_t, local_problem_size, local_problem_size >;
    using f_el_t                      = Eigen::Matrix< val_t, local_problem_size, 1 >;
    return std::pair< k_el_t, f_el_t >{k_el_t::Zero(), f_el_t::Zero()};
}
} // namespace detail

template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalSystem(Kernel&&                                                                       kernel,
                         const Element< ET, EO >&                                                       element,
                         const Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, n_fields >&            node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_q,
                         val_t                                                                          time)
    requires detail::Kernel_c< Kernel, ET, EO, n_fields >
{
    const auto& quadrature          = basis_at_q.quadrature;
    const auto  jac_at_qp           = computeJacobiansAtQpoints(element, quadrature);
    const auto& basis_vals          = basis_at_q.basis_vals;
    const auto  basis_ders          = computePhysBasisDersAtQpoints(basis_at_q.basis_ders, jac_at_qp);
    const auto  field_vals_and_ders = detail::computeFieldValsAndDers(basis_vals, basis_ders, node_vals);

    auto       local_system = detail::initLocalSystem< Kernel, ET, EO, n_fields >();
    const auto process_qp   = [&](ptrdiff_t qp_ind) {
        auto& [K_el, F_el] = local_system;
        const auto [A, F]  = detail::evaluateKernel(kernel, element, field_vals_and_ders, quadrature, qp_ind, time);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, basis_vals, basis_ders, qp_ind);
        const auto rank_update_weight = jac_at_qp[qp_ind].determinant() * quadrature.getWeights()[qp_ind];
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
    };

    for (ptrdiff_t qp_ind = 0; qp_ind < static_cast< ptrdiff_t >(quadrature.size); ++qp_ind)
        process_qp(qp_ind);
    return local_system;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
