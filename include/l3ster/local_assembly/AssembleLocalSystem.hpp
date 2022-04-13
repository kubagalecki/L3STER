#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/local_assembly/ComputePhysBasesAtQpoints.hpp"
#include "l3ster/local_assembly/KernelTraits.hpp"
#include "l3ster/local_assembly/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"

namespace lstr
{
namespace detail
{
template < typename Kernel, ElementTypes ET, el_o_t EO, KernelFieldDependence_c auto field_dep >
auto computeFieldValues(const auto& basis_vals,
                        const auto& node_vals) requires ValidKernel_c< Kernel, ET, EO, field_dep >
{
    constexpr auto n_fields = field_dep.n_fields;
    constexpr auto n_qp     = std::remove_cvref_t< decltype(basis_vals) >::RowsAtCompileTime;
    if constexpr (ConstantKernel_c< Kernel, ET, EO, field_dep >)
        return Eigen::Matrix< val_t, n_qp, 0 >{};
    else
        return Eigen::Matrix< val_t, n_qp, n_fields >{basis_vals * node_vals};
}

template < typename Kernel, ElementTypes ET, el_o_t EO, KernelFieldDependence_c auto field_dep >
auto computeFieldDerivatives(const auto& basis_ders,
                             const auto& node_vals) requires ValidKernel_c< Kernel, ET, EO, field_dep >
{
    constexpr auto  n_ders   = field_dep.der_inds.size();
    constexpr auto  n_qp     = std::remove_cvref_t< decltype(basis_ders) >::value_type::RowsAtCompileTime;
    constexpr auto& der_inds = field_dep.der_inds;
    if constexpr (n_ders > 0 and
                  (FullKernel_c< Kernel, ET, EO, field_dep > or SpaceIndependentKernel_c< Kernel, ET, EO, field_dep >))
    {
        // TODO
        // Note: this is a workaround for an Eigen bug (issue #2375). Once this is fixed, instead of making an explicit
        // copy, an indexed view should be used ( i.e. node_vals(Eigen::all, der_indices) )
        const auto node_vals_for_ders = [&] {
            Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, n_ders > ret_val{};
            for (ptrdiff_t col_ind = 0; col_ind < static_cast< ptrdiff_t >(n_ders); ++col_ind)
                for (ptrdiff_t row = 0; row < static_cast< ptrdiff_t >(Element< ET, EO >::n_nodes); ++row)
                    ret_val(row, col_ind) = node_vals(row, der_inds[col_ind]);
            return ret_val;
        }();

        using der_t = std::array< Eigen::Matrix< val_t, n_qp, n_ders >, Element< ET, EO >::native_dim >;
        der_t ret_val;
        for (ptrdiff_t i = 0; i < Element< ET, EO >::native_dim; ++i)
            ret_val[i] = basis_ders[i] * node_vals_for_ders;
        return ret_val;
    }
    else
        return std::array< Eigen::Matrix< val_t, n_qp, 0 >, Element< ET, EO >::native_dim >{};
}

template < KernelFieldDependence_c auto field_dep >
auto computeFieldValuesAtQpoint(const auto& field_values, ptrdiff_t qp_ind)
{
    std::array< val_t, field_dep.n_fields > ret_val;
    for (ptrdiff_t i = 0; auto& v : ret_val)
        v = field_values(qp_ind, i++);
    return ret_val;
}

template < ElementTypes ET, KernelFieldDependence_c auto field_dep >
auto computeFieldDerivativesAtQpoint(const auto& field_ders, ptrdiff_t qp_ind)
{
    std::array< std::array< val_t, field_dep.der_inds.size() >, Element< ET, 1 >::native_dim > ret_val;
    for (ptrdiff_t dim_ind = 0; auto& dim : ret_val)
    {
        for (ptrdiff_t i = 0; auto& der : dim)
            der = field_ders[dim_ind](qp_ind, i++);
        ++dim_ind;
    }
    return ret_val;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, KernelFieldDependence_c auto field_dep >
auto evaluateKernel(Kernel&&    kernel,
                    const auto& element,
                    const auto& field_values,
                    const auto& field_ders,
                    const auto& quadrature,
                    ptrdiff_t   qp_ind)
{
    if constexpr (FullKernel_c< Kernel, ET, EO, field_dep >)
    {
        const auto field_value_at_qp = computeFieldValuesAtQpoint< field_dep >(field_values, qp_ind);
        const auto field_ders_at_qp  = computeFieldDerivativesAtQpoint< ET, field_dep >(field_ders, qp_ind);
        const auto physical_qp       = mapToPhysicalSpace(element, Point{quadrature.getPoints()[qp_ind]});
        return std::invoke(std::forward< Kernel >(kernel), field_value_at_qp, field_ders_at_qp, physical_qp);
    }
    else if constexpr (SpaceIndependentKernel_c< Kernel, ET, EO, field_dep >)
    {
        const auto field_value_at_qp = computeFieldValuesAtQpoint< field_dep >(field_values, qp_ind);
        const auto field_ders_at_qp  = computeFieldDerivativesAtQpoint< ET, field_dep >(field_ders, qp_ind);
        return std::invoke(std::forward< Kernel >(kernel), field_value_at_qp, field_ders_at_qp);
    }
    else if constexpr (DerivativeIndependentKernel_c< Kernel, ET, EO, field_dep >)
    {
        const auto field_value_at_qp = computeFieldValuesAtQpoint< field_dep >(field_values, qp_ind);
        return std::invoke(std::forward< Kernel >(kernel), field_value_at_qp);
    }
    else
        return std::invoke(std::forward< Kernel >(kernel));
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

template < typename Kernel, ElementTypes ET, el_o_t EO, KernelFieldDependence_c auto field_dep >
requires ValidKernel_c< Kernel, ET, EO, field_dep >
auto initLocalSystem()
{
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns< Kernel, ET, EO, field_dep >;
    using k_el_t                      = Eigen::Matrix< val_t, local_problem_size, local_problem_size >;
    using f_el_t                      = Eigen::Matrix< val_t, local_problem_size, 1 >;
    return std::pair< k_el_t, f_el_t >{k_el_t::Zero(), f_el_t::Zero()};
}
} // namespace detail

template < detail::KernelFieldDependence_c auto field_dep,
           typename Kernel,
           q_l_t        QL,
           dim_t        QD,
           ElementTypes ET,
           el_o_t       EO >
auto assembleLocalSystem(Kernel&&                                                                      kernel,
                         const Element< ET, EO >&                                                      element,
                         const Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, field_dep.n_fields >& node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL, QD >& basis_at_q) requires
    detail::ValidKernel_c< Kernel, ET, EO, field_dep >
{
    const auto& quadrature   = basis_at_q.quadrature;
    const auto& basis_vals   = basis_at_q.basis_vals;
    const auto& basis_ders   = basis_at_q.basis_ders;
    const auto  jac_array    = computeElementJacobiansAtQpoints(element, quadrature);
    const auto  field_values = detail::computeFieldValues< Kernel, ET, EO, field_dep >(basis_vals, node_vals);
    const auto  field_ders   = detail::computeFieldDerivatives< Kernel, ET, EO, field_dep >(basis_ders, node_vals);

    auto local_system  = detail::initLocalSystem< Kernel, ET, EO, field_dep >();
    auto& [K_el, F_el] = local_system;

    for (ptrdiff_t qp_ind = 0; auto weight : quadrature.getWeights())
    {
        const auto [A, F] = detail::evaluateKernel< Kernel, ET, EO, field_dep >(
            kernel, element, field_values, field_ders, quadrature, qp_ind);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, basis_vals, basis_ders, qp_ind);
        const auto rank_update_weight = jac_array[qp_ind].determinant() * weight;
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_matrix * F * rank_update_weight;
        ++qp_ind;
    }

    return local_system;
}

template < detail::KernelFieldDependence_c auto field_dep,
           typename Kernel,
           q_l_t        QL,
           dim_t        QD,
           ElementTypes ET,
           el_o_t       EO >
auto assembleLocalSystem(Kernel&&                                            kernel,
                         const Element< ET, EO >&                            element,
                         const ReferenceBasisAtQuadrature< ET, EO, QL, QD >& basis_at_q) requires
    detail::ConstantKernel_c< Kernel, ET, EO, field_dep >
{
    using empty_t = Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, 0 >;
    return assembleLocalSystem< field_dep >(std::forward< Kernel >(kernel), element, empty_t{}, basis_at_q);
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
