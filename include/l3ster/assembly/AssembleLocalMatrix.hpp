#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALMATRIX_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALMATRIX_HPP

#include "l3ster/assembly/ComputePhysBasesAtQpoints.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"

namespace lstr
{
namespace detail
{
template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
requires(N_DERS <= N_FIELDS) struct LocalKernelValidity
{
    static constexpr bool invocable_with_all =
        std::is_nothrow_invocable_v< Kernel,
                                     std::array< val_t, N_FIELDS >,
                                     std::array< std::array< val_t, N_DERS >, Element< ET, EO >::native_dim >,
                                     Point< 3 > >;
    static constexpr bool invocable_with_vals_and_ders =
        std::is_nothrow_invocable_v< Kernel,
                                     std::array< val_t, N_FIELDS >,
                                     std::array< std::array< val_t, N_DERS >, Element< ET, EO >::native_dim > >;
    static constexpr bool invocable_with_vals = std::is_nothrow_invocable_v< Kernel, std::array< val_t, N_FIELDS > >;
    static constexpr bool invocable_with_none = std::is_nothrow_invocable_v< Kernel >;

    static constexpr bool is_valid_kernel =
        exactlyOneOf(invocable_with_all, invocable_with_none, invocable_with_vals, invocable_with_vals_and_ders);
};

template < typename T >
concept ValidKernelResult_c = pair< T > and array< typename T::first_type > and
                              EigenMatrix_c< typename T::first_type::value_type > and
                              EigenMatrix_c< typename T::second_type > and(T::second_type::ColsAtCompileTime == 1) and
                              (T::first_type::value_type::RowsAtCompileTime == T::second_type::RowsAtCompileTime);

template < typename Kernel, ElementTypes ET, el_o_t EO, auto N_FIELDS, size_t N_DERS >
concept ValidFullKernel_c = requires(Kernel                                                                   kernel,
                                     std::array< val_t, N_FIELDS >                                            node_vals,
                                     std::array< std::array< val_t, N_DERS >, Element< ET, EO >::native_dim > node_ders,
                                     Point< 3 >                                                               point)
{
    {
        std::invoke(kernel, node_vals, node_ders, point)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, ElementTypes ET, el_o_t EO, auto N_FIELDS, size_t N_DERS >
concept ValidSpaceInependentKernel_c =
    requires(Kernel                                                                   kernel,
             std::array< val_t, N_FIELDS >                                            node_vals,
             std::array< std::array< val_t, N_DERS >, Element< ET, EO >::native_dim > node_ders)
{
    {
        std::invoke(kernel, node_vals, node_ders)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, auto N_FIELDS >
concept ValidDerivativeInependentKernel_c = requires(Kernel kernel, std::array< val_t, N_FIELDS > node_vals)
{
    {
        std::invoke(kernel, node_vals)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel >
concept ValidConstantKernel_c = requires(Kernel kernel)
{
    {
        std::invoke(kernel)
    }
    noexcept->ValidKernelResult_c;
};

template < typename Kernel, ElementTypes ET, el_o_t EO, auto N_FIELDS, size_t N_DERS >
concept ValidKernel_c = LocalKernelValidity< Kernel, ET, EO, N_FIELDS, N_DERS >::is_valid_kernel and
    (ValidFullKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS > or
     ValidSpaceInependentKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS > or
     ValidDerivativeInependentKernel_c< Kernel, N_FIELDS > or ValidConstantKernel_c< Kernel >);

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
    auto computeFieldValues(const auto& basis_vals, const auto& node_vals) requires(N_FIELDS > 0) and
    (ValidFullKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS > or
     ValidSpaceInependentKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS > or
     ValidDerivativeInependentKernel_c< Kernel, N_FIELDS >)
{
    constexpr auto n_qp = std::remove_cvref_t< decltype(basis_vals) >::RowsAtCompileTime;
    return Eigen::Matrix< val_t, n_qp, N_FIELDS >{basis_vals * node_vals};
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
    auto computeFieldValues(const auto& basis_vals, [[maybe_unused]] const auto& node_vals) requires(N_FIELDS == 0) or
    ValidConstantKernel_c< Kernel >
{
    constexpr auto n_qp = std::remove_cvref_t< decltype(basis_vals) >::RowsAtCompileTime;
    return Eigen::Matrix< val_t, n_qp, 0 >{};
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
    auto computeFieldDerivatives(const auto& basis_ders,
                                 const auto& node_vals,
                                 const auto& der_indices) requires(N_DERS > 0) and
    (ValidFullKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS > or
     ValidSpaceInependentKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS >)
{
    // TODO
    // Note: this is a workaround for an Eigen bug (issue #2375). Once this is fixed, instead of making an explicit
    // copy, an indexed view should be used ( i.e. node_vals(Eigen::all, der_indices) )
    const auto node_vals_for_ders = [&] {
        Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, N_DERS > ret_val{};
        for (ptrdiff_t col_ind = 0; col_ind < static_cast< ptrdiff_t >(N_DERS); ++col_ind)
            for (ptrdiff_t row = 0; row < static_cast< ptrdiff_t >(Element< ET, EO >::n_nodes); ++row)
                ret_val(row, col_ind) = node_vals(row, der_indices[col_ind]);
        return ret_val;
    }();

    constexpr auto n_qp = std::remove_cvref_t< decltype(basis_ders) >::value_type::RowsAtCompileTime;
    using der_t         = std::array< Eigen::Matrix< val_t, n_qp, N_DERS >, Element< ET, EO >::native_dim >;
    der_t ret_val{};
    for (ptrdiff_t i = 0; i < Element< ET, EO >::native_dim; ++i)
        ret_val[i] = basis_ders[i] * node_vals_for_ders;
    return ret_val;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
    auto computeFieldDerivatives(const auto&                  basis_ders,
                                 [[maybe_unused]] const auto& node_vals,
                                 [[maybe_unused]] const auto& der_indices) requires(N_DERS == 0) or
    ValidDerivativeInependentKernel_c< Kernel, N_FIELDS > or ValidConstantKernel_c< Kernel >
{
    constexpr auto n_qp = std::remove_cvref_t< decltype(basis_ders) >::value_type::RowsAtCompileTime;
    return std::array< Eigen::Matrix< val_t, n_qp, 0 >, Element< ET, EO >::native_dim >{};
}

template < std::integral auto N_FIELDS >
auto computeFieldValuesAtQpoint(const auto& field_values, ptrdiff_t qp_ind)
{
    std::array< val_t, N_FIELDS > ret_val;
    for (ptrdiff_t i = 0; auto& v : ret_val)
        v = field_values(qp_ind, i++);
    return ret_val;
}

template < ElementTypes ET, el_o_t EO, size_t N_DERS >
auto computeFieldDerivativesAtQpoint(const auto& field_ders, ptrdiff_t qp_ind)
{
    std::array< std::array< val_t, N_DERS >, Element< ET, EO >::native_dim > ret_val;
    for (ptrdiff_t dim_ind = 0; auto& dim : ret_val)
    {
        for (ptrdiff_t i = 0; auto& der : dim)
            der = field_ders[dim_ind](qp_ind, i++);
        ++dim_ind;
    }
    return ret_val;
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
auto evaluateKernel(Kernel&&    kernel,
                    const auto& element,
                    const auto& field_values,
                    const auto& field_ders,
                    const auto& quadrature,
                    ptrdiff_t   qp_ind) requires ValidFullKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS >
{
    const auto field_value_at_qp = computeFieldValuesAtQpoint< N_FIELDS >(field_values, qp_ind);
    const auto field_ders_at_qp  = computeFieldDerivativesAtQpoint< ET, EO, N_DERS >(field_ders, qp_ind);
    const auto physical_qp       = mapToPhysicalSpace(element, Point{quadrature.getPoints()[qp_ind]});
    return std::invoke(std::forward< Kernel >(kernel), field_value_at_qp, field_ders_at_qp, physical_qp);
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
auto evaluateKernel(Kernel&&                     kernel,
                    [[maybe_unused]] const auto& element,
                    const auto&                  field_values,
                    const auto&                  field_ders,
                    [[maybe_unused]] const auto& quadrature,
                    ptrdiff_t qp_ind) requires ValidSpaceInependentKernel_c< Kernel, ET, EO, N_FIELDS, N_DERS >
{
    const auto field_value_at_qp = computeFieldValuesAtQpoint< N_FIELDS >(field_values, qp_ind);
    const auto field_ders_at_qp  = computeFieldDerivativesAtQpoint< ET, EO, N_DERS >(field_ders, qp_ind);
    return std::invoke(std::forward< Kernel >(kernel), field_value_at_qp, field_ders_at_qp);
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
auto evaluateKernel(Kernel&&                     kernel,
                    [[maybe_unused]] const auto& element,
                    const auto&                  field_values,
                    [[maybe_unused]] const auto& field_ders,
                    [[maybe_unused]] const auto& quadrature,
                    ptrdiff_t                    qp_ind) requires ValidDerivativeInependentKernel_c< Kernel, N_FIELDS >
{
    const auto field_value_at_qp = computeFieldValuesAtQpoint< N_FIELDS >(field_values, qp_ind);
    return std::invoke(std::forward< Kernel >(kernel), field_value_at_qp);
}

template < typename Kernel, ElementTypes ET, el_o_t EO, std::integral auto N_FIELDS, size_t N_DERS >
auto evaluateKernel(Kernel&&                     kernel,
                    [[maybe_unused]] const auto& element,
                    [[maybe_unused]] const auto& field_values,
                    [[maybe_unused]] const auto& field_ders,
                    [[maybe_unused]] const auto& quadrature,
                    [[maybe_unused]] ptrdiff_t   qp_ind) requires ValidConstantKernel_c< Kernel >
{
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

template < QuadratureTypes QT,
           q_o_t           QO,
           BasisTypes      BT,
           typename Kernel,
           ElementTypes       ET,
           el_o_t             EO,
           std::integral auto N_FIELDS >
auto initLocalSystem(const array auto& der_indices)
{
    constexpr auto n_ders = std::tuple_size_v< std::remove_cvref_t< decltype(der_indices) > >;
    using kernel_ret_t    = std::remove_cvref_t< decltype(
        [&](auto&& kernel, const auto& element, const auto& node_vals) {
            const auto& quadrature = getQuadrature< QT, QO, ET >();
            const auto& basis_vals = getRefBasesAtQpoints< QT, QO, ET, EO, BT >();
            const auto  jac_array  = computeElementJacobiansAtQpoints< QT, QO >(element);
            const auto  basis_ders = computePhysicalBasesAtQpoints< QT, QO, BT, ET, EO >(jac_array);

            const auto field_values =
                detail::computeFieldValues< Kernel, ET, EO, N_FIELDS, n_ders >(basis_vals, node_vals);
            const auto field_ders =
                detail::computeFieldDerivatives< Kernel, ET, EO, N_FIELDS, n_ders >(basis_ders, node_vals, der_indices);

            return detail::evaluateKernel< Kernel, ET, EO, N_FIELDS, n_ders >(
                kernel, element, field_values, field_ders, quadrature, 0);
        }(std::declval< Kernel >(),
          std::declval< Element< ET, EO > >(),
          std::declval< Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, N_FIELDS > >())) >;

    constexpr auto n_unknowns         = kernel_ret_t::first_type::value_type::ColsAtCompileTime;
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns;

    using k_el_t = Eigen::Matrix< val_t, local_problem_size, local_problem_size >;
    using f_el_t = Eigen::Matrix< val_t, local_problem_size, 1 >;
    return std::pair< k_el_t, f_el_t >{k_el_t::Zero(), f_el_t::Zero()};
}
} // namespace detail

template < QuadratureTypes QT,
           q_o_t           QO,
           BasisTypes      BT,
           typename Kernel,
           ElementTypes       ET,
           el_o_t             EO,
           std::integral auto N_FIELDS >
auto assembleLocalMatrix(Kernel&&                                                            kernel,
                         const Element< ET, EO >&                                            element,
                         const Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, N_FIELDS >& node_vals,
                         const array auto&                                                   der_indices) requires
    detail::ValidKernel_c< Kernel, ET, EO, N_FIELDS, std::tuple_size_v< std::remove_cvref_t< decltype(der_indices) > > >
{
    constexpr auto n_ders = std::tuple_size_v< std::remove_cvref_t< decltype(der_indices) > >;

    const auto& quadrature = getQuadrature< QT, QO, ET >();
    const auto& basis_vals = getRefBasesAtQpoints< QT, QO, ET, EO, BT >();
    const auto  jac_array  = computeElementJacobiansAtQpoints< QT, QO >(element);
    const auto  basis_ders = computePhysicalBasesAtQpoints< QT, QO, BT, ET, EO >(jac_array);

    const auto field_values = detail::computeFieldValues< Kernel, ET, EO, N_FIELDS, n_ders >(basis_vals, node_vals);
    const auto field_ders =
        detail::computeFieldDerivatives< Kernel, ET, EO, N_FIELDS, n_ders >(basis_ders, node_vals, der_indices);

    auto local_system  = detail::initLocalSystem< QT, QO, BT, Kernel, ET, EO, N_FIELDS >(der_indices);
    auto& [K_el, F_el] = local_system;

    for (ptrdiff_t qp_ind = 0; auto weight : quadrature.getWeights())
    {
        const auto [A, F] = detail::evaluateKernel< Kernel, ET, EO, N_FIELDS, n_ders >(
            kernel, element, field_values, field_ders, quadrature, qp_ind);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, basis_vals, basis_ders, qp_ind);
        const auto rank_update_weight = jac_array[qp_ind].determinant() * weight;
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_weight * rank_update_matrix * F;
        ++qp_ind;
    }

    return local_system;
}

template < QuadratureTypes QT, q_o_t QO, BasisTypes BT, typename Kernel, ElementTypes ET, el_o_t EO, auto N_FIELDS >
auto assembleLocalMatrix(Kernel&&                                                            kernel,
                         const Element< ET, EO >&                                            element,
                         const Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, N_FIELDS >& node_vals)
{
    return assembleLocalMatrix< QT, QO, BT >(
        std::forward< Kernel >(kernel), element, node_vals, std::array< ptrdiff_t, 0 >{});
}

template < QuadratureTypes QT, q_o_t QO, BasisTypes BT, typename Kernel, ElementTypes ET, el_o_t EO >
auto assembleLocalMatrix(Kernel&& kernel, const Element< ET, EO >& element)
{
    using empty_t = Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, 0 >;
    return assembleLocalMatrix< QT, QO, BT >(std::forward< Kernel >(kernel), element, empty_t{});
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALMATRIX_HPP
