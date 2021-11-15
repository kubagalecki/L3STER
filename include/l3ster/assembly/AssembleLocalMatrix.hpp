#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALMATRIX_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALMATRIX_HPP

#include "l3ster/assembly/ComputePhysBasesAtQpoints.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"

namespace lstr
{
template < QuadratureTypes QT, q_o_t QO, BasisTypes BT, typename Kernel, ElementTypes ET, el_o_t EO, auto N_FIELDS >
auto assembleLocalMatrix(
    Kernel&&                                                            kernel,
    const Element< ET, EO >&                                            element,
    const Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, N_FIELDS >& node_vals,
    const array auto
        der_indices) requires(Element< ET, EO >::native_dim >= std::tuple_size_v< decltype(der_indices) > and
                              std::invocable<
                                  Kernel,
                                  std::array< val_t, N_FIELDS >,
                                  std::array< std::array< val_t, std::tuple_size_v< decltype(der_indices) > >,
                                              Element< ET, EO >::native_dim >,
                                  Point< 3 > >)
{
    // Note: The kernel should return a pair of array {A0, A1, ...} and F, see 5 lines below
    constexpr ptrdiff_t n_inds      = std::tuple_size_v< decltype(der_indices) >;
    using field_val_at_qp_t         = std::array< val_t, N_FIELDS >;
    using single_der_at_qp_t        = std::array< val_t, n_inds >;
    using field_ders_at_qp_t        = std::array< single_der_at_qp_t, Element< ET, EO >::native_dim >;
    using kernel_ret_t              = std::invoke_result_t< Kernel, field_val_at_qp_t, field_ders_at_qp_t, Point< 3 > >;
    using A_t                       = kernel_ret_t::first_type::value_type;
    constexpr size_t    n_unknowns  = A_t::ColsAtCompileTime;
    constexpr size_t    n_equations = A_t::RowsAtCompileTime;
    constexpr ptrdiff_t n_bases     = element.n_nodes;
    constexpr size_t    local_mat_size = n_unknowns * n_bases;

    const auto& quadratre  = getQuadrature< QT, QO, ET >();
    const auto& basis_vals = getRefBasesAtQpoints< QT, QO, ET, EO, BT >();
    const auto  jac_array  = computeElementJacobiansAtQpoints< QT, QO >(element);
    const auto  basis_ders = computePhysicalBasesAtQpoints< QT, QO, BT, ET, EO >(jac_array);

    const auto& [field_values, field_ders] = [&] {
        // TODO
        // Note: this is a workaround for an Eigen bug (issue #2375). Once this is fixed, instead of making an explicit
        // copy, an indexed view should be used ( i.e. node_vals(Eigen::all, der_indices) )
        const auto node_vals_for_ders = [&] {
            Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, n_inds > ret_val;
            for (ptrdiff_t col_ind = 0; col_ind < n_inds; ++col_ind)
                for (ptrdiff_t row = 0; row < static_cast< ptrdiff_t >(Element< ET, EO >::n_nodes); ++row)
                    ret_val(row, col_ind) = node_vals(row, der_indices[col_ind]);
            return ret_val;
        }();

        using value_t = Eigen::Matrix< val_t, quadratre.size, N_FIELDS >;
        using der_t   = std::array< Eigen::Matrix< val_t, quadratre.size, n_inds >, Element< ET, EO >::native_dim >;
        std::pair< value_t, der_t > ret_val; // NOLINT initilization follows directly below
        ret_val.first = basis_vals * node_vals;
        for (ptrdiff_t i = 0; i < Element< ET, EO >::native_dim; ++i)
            ret_val.second[i] = basis_ders[i] * node_vals_for_ders;
        return ret_val;
    }();

    const auto evaluate_kernel = [&](ptrdiff_t qp_ind) {
        const auto field_value_at_qp = [&] {
            field_val_at_qp_t ret_val;
            for (ptrdiff_t i = 0; auto& v : ret_val)
                v = field_values(qp_ind, i++);
            return ret_val;
        }();
        const auto field_ders_at_qp = [&] {
            field_ders_at_qp_t ret_val;
            for (ptrdiff_t dim_ind = 0; auto& dim : ret_val)
            {
                for (ptrdiff_t i = 0; auto& der : dim)
                    der = field_ders[dim_ind](qp_ind, i++);
                ++dim_ind;
            }
            return ret_val;
        }();
        const auto physical_qp = mapToPhysicalSpace< BT >(element, Point{quadratre.getPoints()[qp_ind]});
        return std::invoke(kernel, field_value_at_qp, field_ders_at_qp, physical_qp);
    };

    const auto make_rank_update_matrix = [&](ptrdiff_t qp_ind, const auto& A_ker) {
        using ret_t = Eigen::Matrix< val_t, local_mat_size, n_equations, Eigen::RowMajor >;
        ret_t ret_val; // NOLINT filled in directly below
        for (ptrdiff_t basis_ind = 0; basis_ind < n_bases; ++basis_ind)
        {
            ret_val(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) =
                basis_vals(qp_ind, basis_ind) * A_ker[0].transpose();
            for (ptrdiff_t d = 0; d < Element< ET, EO >::native_dim; ++d)
                ret_val(Eigen::seqN(basis_ind * n_unknowns, Eigen::fix< n_unknowns >), Eigen::all) +=
                    basis_ders[d](qp_ind, basis_ind) * A_ker[d + 1].transpose();
        }
        return ret_val;
    };

    using k_el_t = Eigen::Matrix< val_t, local_mat_size, local_mat_size >;
    using f_el_t = Eigen::Matrix< val_t, local_mat_size, 1 >;
    std::pair< k_el_t, f_el_t > ret_val{k_el_t::Zero(), f_el_t::Zero()};
    auto& [K_el, F_el] = ret_val;

    for (ptrdiff_t qp_ind = 0; auto weight : quadratre.getWeights())
    {
        const auto& [A_ker, F_ker]    = evaluate_kernel(qp_ind);
        const auto rank_update_matrix = make_rank_update_matrix(qp_ind, A_ker);
        const auto rank_update_weight = jac_array[qp_ind].determinant() * weight;
        K_el.template selfadjointView< Eigen::Lower >().rankUpdate(rank_update_matrix, rank_update_weight);
        F_el += rank_update_weight * rank_update_matrix * F_ker;
        ++qp_ind;
    }

    return ret_val;
}
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALMATRIX_HPP
