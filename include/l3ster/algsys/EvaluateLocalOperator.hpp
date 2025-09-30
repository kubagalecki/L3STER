#ifndef L3STER_ALGSYS_EVALUATELOCALOPERATOR
#define L3STER_ALGSYS_EVALUATELOCALOPERATOR

#include "l3ster/algsys/AssembleLocalSystem.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/util/CacheSizesAtCompileTime.hpp"

#include <numeric>
#include <optional>

namespace lstr::algsys
{
namespace detail
{
template < int R, int C, int Maj, size_t N >
auto transposeOperators(const std::array< Eigen::Matrix< val_t, R, C, Maj >, N >& operators)
    -> std::array< Eigen::Matrix< val_t, C, R, Eigen::ColMajor >, N >
{
    auto retval = std::array< Eigen::Matrix< val_t, C, R, Eigen::ColMajor >, N >{};
    for (auto&& [ret, op] : std::views::zip(retval, operators))
        ret = op.transpose();
    return retval;
}

template < int R, int C, size_t N >
auto computeATrans(const std::array< Eigen::Matrix< val_t, R, C, Eigen::ColMajor >, N >& operators_trans,
                   val_t                                                                 bas_val,
                   Eigen::Vector< val_t, N - 1 > der_vals) -> Eigen::Matrix< val_t, R, C, Eigen::ColMajor >
{
    Eigen::Matrix< val_t, R, C, Eigen::ColMajor > retval = operators_trans.front() * bas_val;
    for (auto&& [Ai, di] : std::views::zip(operators_trans | std::views::drop(1), der_vals))
        retval += Ai * di;
    return retval;
}
} // namespace detail

template < size_t operator_size, size_t update_size, size_t n_rhs >
class OperatorEvaluationManager
{
    static consteval size_t batchSizeHeuristic()
    {
        constexpr size_t ilp_fma       = 2;
        constexpr size_t min_cols      = ilp_fma * util::simd_width / sizeof(val_t);
        const auto       try_fit_cache = [&](size_t cache_bytes) -> std::optional< size_t > {
            const size_t cache_capacity = cache_bytes / sizeof(val_t) - operator_size;
            const size_t fitting_cols   = cache_capacity / operator_size;
            if (min_cols * update_size <= fitting_cols)
                return {min_cols};
            const size_t batches = fitting_cols / update_size;
            if (batches > 0 and batches * update_size >= min_cols)
                return {batches};
            else
                return std::nullopt;
        };
        auto retval = std::optional< size_t >{};
        for (auto cache_size : util::cache_sizes)
            retval = retval.or_else([&] { return try_fit_cache(cache_size); });
        return retval.value_or(std::lcm(min_cols, update_size) / update_size);
    }
    static constexpr size_t updates_per_batch = batchSizeHeuristic();
    static constexpr size_t batch_update_size = update_size * updates_per_batch;
    using batch_update_matrix_t = Eigen::Matrix< val_t, operator_size, batch_update_size, Eigen::ColMajor >;
    using operand_t             = util::eigen::MatrixMaxCol_t< val_t, operator_size, n_rhs >;

public:
    template < int n_unknowns, int n_bases, int dim >
    void update(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
                const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                const util::eigen::RowMajorMatrix< val_t, dim, n_bases >&                     basis_ders,
                val_t                                                                         weight,
                const operand_t&                                                              x,
                operand_t&                                                                    y)
    {
        if (fillBatch(kernel_result, basis_vals, basis_ders, weight))
            flushFull(x, y);
    }
    void finalize(const operand_t& x, operand_t& y) { flush(x, y); }

private:
    template < int n_unknowns, int n_bases, int dim >
    bool fillBatch(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
                   const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                   const util::eigen::RowMajorMatrix< val_t, dim, n_bases >&                     basis_ders,
                   val_t                                                                         weight);
    void flushImpl(auto&& update_block, const operand_t& x, operand_t& y);
    void flushFull(const operand_t& x, operand_t& y) { flushImpl(*m_update_matrix, x, y); }
    void flush(const operand_t& x, operand_t& y) { flushImpl(m_update_matrix->leftCols(m_filled_cols), x, y); }

    std::unique_ptr< batch_update_matrix_t >  m_update_matrix = std::make_unique< batch_update_matrix_t >();
    Eigen::Vector< val_t, batch_update_size > m_weights       = {};
    int                                       m_filled_cols   = 0;
};

template < size_t operator_size, size_t update_size, size_t n_rhs >
template < int n_unknowns, int n_bases, int dim >
bool OperatorEvaluationManager< operator_size, update_size, n_rhs >::fillBatch(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
    const Eigen::Vector< lstr::val_t, n_bases >&                                  basis_vals,
    const util::eigen::RowMajorMatrix< lstr::val_t, dim, n_bases >&               basis_ders,
    val_t                                                                         weight)
{
    // Note: this function is performance critical
    // Filling the batch matrix column by column allows the compiler to keep the A^T columns in registers

    L3STER_PROFILE_FUNCTION;
    const auto As_trans = detail::transposeOperators(kernel_result);
    using col_t         = Eigen::Vector< val_t, n_unknowns >;
    for (int eq = 0; eq != update_size; ++eq)
    {
        const auto  dest_col = m_filled_cols + eq;
        const col_t val_col  = As_trans.front().col(eq);
        auto        der_cols = std::array< col_t, dim >{};
        for (int i = 0; i != dim; ++i)
            der_cols[i] = As_trans[i + 1].col(eq);
        for (int basis = 0; basis != n_bases; ++basis)
        {
            const auto dest_row = basis * n_unknowns;
            col_t      col      = val_col * basis_vals[basis];
            for (int d = 0; d != dim; ++d)
                col += der_cols[d] * basis_ders(d, basis);
            m_update_matrix->template block< n_unknowns, 1 >(dest_row, dest_col) = col;
        }
    }
    m_weights.template segment< update_size >(m_filled_cols).setConstant(weight);
    m_filled_cols += update_size;
    return m_filled_cols == batch_update_size;
}

template < size_t operator_size, size_t update_size, size_t n_rhs >
void OperatorEvaluationManager< operator_size, update_size, n_rhs >::flushImpl(
    auto&& update_block, const OperatorEvaluationManager::operand_t& x, OperatorEvaluationManager::operand_t& y)
{
    L3STER_PROFILE_FUNCTION;
    constexpr int intermediate_size = std::remove_cvref_t< decltype(update_block) >::ColsAtCompileTime;
    using A_times_x_t               = Eigen::Matrix< val_t, intermediate_size, 1, Eigen::ColMajor, batch_update_size >;

    // Multiple gemv calls are more efficient than 1 "thin" gemm call
    for (int rhs = 0; rhs != x.cols(); ++rhs)
    {
        auto At_times_x = A_times_x_t{update_block.transpose() * x.col(rhs)};
        for (int i = 0; i != At_times_x.rows(); ++i)
            At_times_x[i] *= m_weights[i];
        y.col(rhs) += update_block * At_times_x;
    }
    m_filled_cols = 0;
}

template < mesh::ElementType ET, el_o_t EO, KernelParams params >
inline constexpr size_t operand_size = mesh::Element< ET, EO >::n_nodes * params.n_unknowns;
template < mesh::ElementType ET, el_o_t EO, KernelParams params >
using Operand = util::eigen::MatrixMaxCol_t< val_t, operand_size< ET, EO, params >, params.n_rhs >;
template < mesh::ElementType ET, el_o_t EO, KernelParams params >
using DirichletInds = std::span< const util::smallest_integral_t< operand_size< ET, EO, params > > >;
template < mesh::ElementType ET, el_o_t EO, KernelParams params >
using DirichletVals =
    Eigen::Matrix< val_t, Eigen::Dynamic, params.n_rhs, Eigen::ColMajor, operand_size< ET, EO, params > >;

template < mesh::ElementType ET, el_o_t EO, KernelParams params >
auto& getLocalOperatorEvalManager()
{
    constexpr auto operand_size = static_cast< size_t >(Operand< ET, EO, params >::RowsAtCompileTime);
    static_assert(operand_size > 0);
    constexpr auto num_equations = params.n_equations;
    constexpr auto num_rhs       = params.n_rhs;
    using eval_manager_t         = OperatorEvaluationManager< operand_size, num_equations, num_rhs >;
    auto& retval                 = util::getThreadLocal< eval_manager_t >();
    return retval;
}

namespace detail
{
template < KernelParams params, int n_bases, std::integral I >
void precomputeDiagRhsImpl(const typename KernelInterface< params >::Result&                      kernel_result,
                           const Eigen::Vector< val_t, n_bases >&                                 basis_vals,
                           const util::eigen::RowMajorMatrix< val_t, params.dimension, n_bases >& basis_ders,
                           val_t                                                                  weight,
                           Eigen::Vector< val_t, n_bases * params.n_unknowns >&                   diagonal,
                           Eigen::Matrix< val_t, n_bases * params.n_unknowns, params.n_rhs >&     rhs,
                           std::span< const I >                                                   dirichlet_inds,
                           const util::eigen::Matrix_c auto&                                      dirichlet_vals)
{
    constexpr auto nukn         = static_cast< int >(params.n_unknowns);
    constexpr auto neq          = static_cast< int >(params.n_equations);
    const auto     ops_trans    = detail::transposeOperators(kernel_result.operators);
    const auto     update_block = [&](const auto& At, int basis) {
        diagonal.template segment< nukn >(basis * nukn) += At.rowwise().squaredNorm() * weight;
        rhs.template middleRows< nukn >(basis * nukn) += At * kernel_result.rhs * weight;
    };
    if (dirichlet_inds.empty())
        for (int basis = 0; basis != n_bases; ++basis)
        {
            const auto At = detail::computeATrans(ops_trans, basis_vals[basis], basis_ders.col(basis));
            update_block(At, basis);
        }
    else
    {
        auto update_mat = Eigen::Matrix< val_t, nukn * n_bases, neq >{};
        for (int basis = 0; basis != n_bases; ++basis)
        {
            const auto At = detail::computeATrans(ops_trans, basis_vals[basis], basis_ders.col(basis));
            update_block(At, basis);
            update_mat.template block< nukn, neq >(basis * nukn, 0) = At;
        }
        Eigen::Matrix< val_t, neq, params.n_rhs > intermediate =
            update_mat(dirichlet_inds, Eigen::all).transpose() * dirichlet_vals * weight;
        rhs -= update_mat * intermediate;
    }
}
} // namespace detail

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
auto evaluateLocalOperator(
    const DomainEquationKernel< Kernel, params >&                                                  kernel,
    const mesh::LocalElementView< ET, EO >&                                                        element,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time,
    const Operand< ET, EO, params >& x) -> Operand< ET, EO, params >
{
    L3STER_PROFILE_FUNCTION;
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);
    Operand< ET, EO, params > y(x.rows(), x.cols());
    const auto&               el_data              = element.getData();
    const auto                jacobi_mat_generator = map::getNatJacobiMatGenerator(el_data);
    auto&                     eval_manager         = getLocalOperatorEvalManager< ET, EO, params >();
    const auto process_qp = [&](auto point, val_t weight, const auto& basis_vals, const auto& ref_basis_ders) {
        const auto [phys_ders, jacobian] = map::mapDomain< ET, EO >(jacobi_mat_generator, point, ref_basis_ders);
        const auto [A, _]                = evalKernel(kernel, point, basis_vals, phys_ders, node_vals, el_data, time);
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        eval_manager.update(A, basis_vals, phys_ders, jacobian * weight, x, y);
    };
    y.setZero();
    basis_at_qps.forEach(process_qp);
    eval_manager.finalize(x, y);
    return y;
}

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
auto evaluateLocalOperator(
    const BoundaryEquationKernel< Kernel, params >&                                                kernel,
    const mesh::LocalElementBoundaryView< ET, EO >&                                                el_view,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time,
    const Operand< ET, EO, params >& x) -> Operand< ET, EO, params >
{
    L3STER_PROFILE_FUNCTION;
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);
    Operand< ET, EO, params > y(x.rows(), x.cols());
    const auto                el_data      = el_view->getData();
    const auto                jacobi_gen   = map::getNatJacobiMatGenerator(el_data);
    auto&                     eval_manager = getLocalOperatorEvalManager< ET, EO, params >();
    const auto                side         = el_view.getSide();
    const auto process_qp = [&](auto point, val_t weight, const auto& basis_vals, const auto& ref_basis_ders) {
        const auto [phys_ders, jacobian, normal] = map::mapBoundary< ET, EO >(jacobi_gen, point, ref_basis_ders, side);
        const auto [A, _] = evalKernel(kernel, point, basis_vals, phys_ders, node_vals, el_data, time, normal);
        eval_manager.update(A, basis_vals, phys_ders, jacobian * weight, x, y);
    };
    y.setZero();
    basis_at_qps.forEach(process_qp);
    eval_manager.finalize(x, y);
    return y;
}

template < KernelParams params, mesh::ElementType ET, el_o_t EO >
struct InitResult
{
    static constexpr size_t size = mesh::Element< ET, EO >::n_nodes * params.n_unknowns;
    using diagonal_t             = Eigen::Vector< val_t, size >;
    using rhs_t                  = Eigen::Matrix< val_t, size, params.n_rhs >;

    diagonal_t diagonal = diagonal_t::Zero();
    rhs_t      rhs      = rhs_t::Zero();
};

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
auto precomputeOperatorDiagonalAndRhs(
    const DomainEquationKernel< Kernel, params >&                                                  kernel,
    const mesh::LocalElementView< ET, EO >&                                                        element,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time,
    DirichletInds< ET, EO, params >                                                                dirichlet_inds,
    const DirichletVals< ET, EO, params >& dirichlet_vals) -> InitResult< params, ET, EO >
{
    L3STER_PROFILE_FUNCTION;
    auto retval           = InitResult< params, ET, EO >{};
    auto& [diagonal, rhs] = retval;
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);
    const auto& el_data              = element.getData();
    const auto  jacobi_mat_generator = map::getNatJacobiMatGenerator(el_data);
    const auto  process_qp = [&](auto point, val_t weight, const auto& basis_vals, const auto& ref_basis_ders) {
        const auto [phys_ders, jacobian] = map::mapDomain< ET, EO >(jacobi_mat_generator, point, ref_basis_ders);
        const auto kernel_result         = evalKernel(kernel, point, basis_vals, phys_ders, node_vals, el_data, time);
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        detail::precomputeDiagRhsImpl< params >(
            kernel_result, basis_vals, phys_ders, jacobian * weight, diagonal, rhs, dirichlet_inds, dirichlet_vals);
    };
    basis_at_qps.forEach(process_qp);
    return retval;
}

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
auto precomputeOperatorDiagonalAndRhs(
    const BoundaryEquationKernel< Kernel, params >&                                                kernel,
    const mesh::LocalElementBoundaryView< ET, EO >&                                                el_view,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time,
    DirichletInds< ET, EO, params >                                                                dirichlet_inds,
    const DirichletVals< ET, EO, params >& dirichlet_vals) -> InitResult< params, ET, EO >
{
    L3STER_PROFILE_FUNCTION;
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);
    auto retval            = InitResult< params, ET, EO >{};
    auto& [diagonal, rhs]  = retval;
    const auto& el_data    = el_view->getData();
    const auto  jacobi_gen = map::getNatJacobiMatGenerator(el_data);
    const auto  side       = el_view.getSide();
    const auto  process_qp = [&](auto point, val_t weight, const auto& basis_vals, const auto& ref_basis_ders) {
        const auto [phys_ders, jacobian, normal] = map::mapBoundary< ET, EO >(jacobi_gen, point, ref_basis_ders, side);
        const auto kernel_result = evalKernel(kernel, point, basis_vals, phys_ders, node_vals, el_data, time, normal);
        detail::precomputeDiagRhsImpl< params >(
            kernel_result, basis_vals, phys_ders, jacobian * weight, diagonal, rhs, dirichlet_inds, dirichlet_vals);
    };
    basis_at_qps.forEach(process_qp);
    return retval;
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_EVALUATELOCALOPERATOR
