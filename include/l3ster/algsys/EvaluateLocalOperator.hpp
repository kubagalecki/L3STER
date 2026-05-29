#ifndef L3STER_ALGSYS_EVALUATELOCALOPERATOR
#define L3STER_ALGSYS_EVALUATELOCALOPERATOR

#include "l3ster/algsys/AssembleLocalSystem.hpp"
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
    template < int n_unknowns, size_t dimp1, typename Bvals, typename Bders >
    void update(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
                const Bvals&                                                                basis_vals,
                const Bders&                                                                basis_ders,
                val_t                                                                       weight,
                const operand_t&                                                            x,
                operand_t&                                                                  y)
    {
        if (fillBatch(kernel_result, basis_vals, basis_ders, weight))
            flushFull(x, y);
    }
    void finalize(const operand_t& x, operand_t& y) { flush(x, y); }

private:
    template < int n_unknowns, size_t dimp1, typename Bvals, typename Bders >
    bool fillBatch(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
                   const Bvals&                                                                basis_vals,
                   const Bders&                                                                basis_ders,
                   val_t                                                                       weight);
    void flushImpl(auto&& update_block, const operand_t& x, operand_t& y);
    void flushFull(const operand_t& x, operand_t& y) { flushImpl(*m_update_matrix, x, y); }
    void flush(const operand_t& x, operand_t& y) { flushImpl(m_update_matrix->leftCols(m_filled_cols), x, y); }

    std::unique_ptr< batch_update_matrix_t >  m_update_matrix = std::make_unique< batch_update_matrix_t >();
    Eigen::Vector< val_t, batch_update_size > m_weights       = {};
    int                                       m_filled_cols   = 0;
};

template < size_t operator_size, size_t update_size, size_t n_rhs >
template < int n_unknowns, size_t dimp1, typename Bvals, typename Bders >
bool OperatorEvaluationManager< operator_size, update_size, n_rhs >::fillBatch(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
    const Bvals&                                                                basis_vals,
    const Bders&                                                                basis_ders,
    val_t                                                                       weight)
{
    // Note: this function is performance-critical
    // Filling the batch matrix column by column allows the compiler to keep the A^T columns in registers

    L3STER_PROFILE_FUNCTION;
    constexpr int dim      = dimp1 - 1;
    constexpr int n_bases  = Bders::RowsAtCompileTime;
    const auto    As_trans = detail::transposeOperators(kernel_result);
    using col_t            = Eigen::Vector< val_t, n_unknowns >;
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
                col += der_cols[d] * basis_ders(basis, d);
            m_update_matrix->template block< n_unknowns, 1 >(dest_row, dest_col) = col;
        }
    }
    m_weights.template segment< update_size >(m_filled_cols).setConstant(weight);
    m_filled_cols += update_size;
    return m_filled_cols == batch_update_size;
}

template < size_t operator_size, size_t update_size, size_t n_rhs >
void OperatorEvaluationManager< operator_size, update_size, n_rhs >::flushImpl(auto&&           update_block,
                                                                               const operand_t& x,
                                                                               operand_t&       y)
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

template < KernelParams params, size_t NB >
inline constexpr size_t operand_size = NB * params.n_unknowns;
template < KernelParams params, size_t NB >
using Operand = util::eigen::MatrixMaxCol_t< val_t, operand_size< params, NB >, params.n_rhs >;
template < KernelParams params, size_t NB >
using DirichletInds = std::span< const util::smallest_integral_t< operand_size< params, NB > > >;
template < KernelParams params, size_t NB >
using DirichletVals = Eigen::Matrix< val_t, Eigen::Dynamic, params.n_rhs, Eigen::ColMajor, operand_size< params, NB > >;

template < KernelParams params, size_t NB >
auto& getLocalOperatorEvalManager()
{
    constexpr auto operand_size = static_cast< size_t >(Operand< params, NB >::RowsAtCompileTime);
    static_assert(operand_size > 0);
    constexpr auto num_equations = params.n_equations;
    constexpr auto num_rhs       = params.n_rhs;
    using eval_manager_t         = OperatorEvaluationManager< operand_size, num_equations, num_rhs >;
    auto& retval                 = util::getThreadLocal< eval_manager_t >();
    return retval;
}

namespace detail
{
template < KernelParams params, int total_unknowns, typename Bvals, typename Bders, typename Dind, typename Dvals >
void precomputeDiagRhsImpl(const typename KernelInterface< params >::Result&     kernel_result,
                           const Bvals&                                          basis_vals,
                           const Bders&                                          basis_ders,
                           val_t                                                 weight,
                           Eigen::Vector< val_t, total_unknowns >&               diagonal,
                           Eigen::Matrix< val_t, total_unknowns, params.n_rhs >& rhs,
                           std::span< const Dind >                               dirichlet_inds,
                           const Dvals&                                          dirichlet_vals)
{
    constexpr auto nukn         = static_cast< int >(params.n_unknowns);
    constexpr auto neq          = static_cast< int >(params.n_equations);
    constexpr int  n_bases      = total_unknowns / nukn;
    const auto     ops_trans    = detail::transposeOperators(kernel_result.operators);
    const auto     update_block = [&](const auto& At, int basis) {
        diagonal.template segment< nukn >(basis * nukn) += At.rowwise().squaredNorm() * weight;
        rhs.template middleRows< nukn >(basis * nukn) += At * kernel_result.rhs * weight;
    };
    if (dirichlet_inds.empty())
        for (int basis = 0; basis != n_bases; ++basis)
        {
            const auto At = detail::computeATrans(ops_trans, basis_vals[basis], basis_ders.row(basis).transpose());
            update_block(At, basis);
        }
    else
    {
        auto update_mat = Eigen::Matrix< val_t, nukn * n_bases, neq >{};
        for (int basis = 0; basis != n_bases; ++basis)
        {
            const auto At = detail::computeATrans(ops_trans, basis_vals[basis], basis_ders.row(basis).transpose());
            update_block(At, basis);
            update_mat.template block< nukn, neq >(basis * nukn, 0) = At;
        }
        Eigen::Matrix< val_t, neq, params.n_rhs > intermediate =
            update_mat(dirichlet_inds, Eigen::all).transpose() * dirichlet_vals * weight;
        rhs -= update_mat * intermediate;
    }
}
} // namespace detail

template < typename Kernel, KernelParams params, size_t NB, size_t NP >
auto evaluateLocalOperator(const DomainEquationKernel< Kernel, params >&                            kernel,
                           const map::TabulatedDomainMapping< NB, NP, params.dimension >&           mapping,
                           const map::FieldValuesAtPoints< NP, params.n_fields, params.dimension >& fields,
                           std::span< const val_t, NP >                                             quad_wgt,
                           val_t                                                                    time,
                           const Operand< params, NB >& x) -> Operand< params, NB >
{
    L3STER_PROFILE_FUNCTION;
    const auto& [J, basis_vals, basis_ders, points] = mapping;
    auto y                                          = Operand< params, NB >(x.rows(), x.cols());
    y.setZero();
    auto& eval_manager = getLocalOperatorEvalManager< params, NB >();
    for (size_t p = 0; p != NP; ++p)
    {
        const auto point    = SpaceTimePoint{points[p], time};
        const auto [fv, fd] = fields.get(p);
        const auto [A, _]   = kernel({fv, fd, point});
        const auto weight   = J[p] * quad_wgt[p];
        eval_manager.update(A, basis_vals->getMap().row(p), basis_ders.getPointMap(p), weight, x, y);
    }
    eval_manager.finalize(x, y);
    return y;
}

template < typename Kernel, KernelParams params, size_t NB, size_t NP >
auto evaluateLocalOperator(const BoundaryEquationKernel< Kernel, params >&                          kernel,
                           const map::TabulatedBoundaryMapping< NB, NP, params.dimension >&         mapping,
                           const map::FieldValuesAtPoints< NP, params.n_fields, params.dimension >& fields,
                           std::span< const val_t, NP >                                             quad_wgt,
                           val_t                                                                    time,
                           const Operand< params, NB >& x) -> Operand< params, NB >
{
    L3STER_PROFILE_FUNCTION;
    const auto& [J, basis_vals, basis_ders, points, normals] = mapping;
    auto y                                                   = Operand< params, NB >(x.rows(), x.cols());
    y.setZero();
    auto& eval_manager = getLocalOperatorEvalManager< params, NB >();
    for (size_t p = 0; p != NP; ++p)
    {
        const auto point    = SpaceTimePoint{points[p], time};
        const auto [fv, fd] = fields.get(p);
        const auto [A, _]   = kernel({fv, fd, point, normals[p]});
        const auto weight   = J[p] * quad_wgt[p];
        eval_manager.update(A, basis_vals->getMap().row(p), basis_ders.getPointMap(p), weight, x, y);
    }
    eval_manager.finalize(x, y);
    return y;
}

template < KernelParams params, size_t NB >
struct InitResult
{
    static constexpr size_t size = NB * params.n_unknowns;
    using diagonal_t             = Eigen::Vector< val_t, size >;
    using rhs_t                  = Eigen::Matrix< val_t, size, params.n_rhs >;

    diagonal_t diagonal = diagonal_t::Zero();
    rhs_t      rhs      = rhs_t::Zero();
};

template < typename Kernel, KernelParams params, size_t NB, size_t NP >
auto precomputeOperatorDiagonalAndRhs(const DomainEquationKernel< Kernel, params >&                            kernel,
                                      const map::TabulatedDomainMapping< NB, NP, params.dimension >&           mapping,
                                      const map::FieldValuesAtPoints< NP, params.n_fields, params.dimension >& fields,
                                      std::span< const val_t, NP >                                             quad_wgt,
                                      val_t                                                                    time,
                                      DirichletInds< params, NB >                                              dir_inds,
                                      const DirichletVals< params, NB >& dir_vals) -> InitResult< params, NB >
{
    L3STER_PROFILE_FUNCTION;
    const auto& [J, basis_vals, basis_ders, points] = mapping;
    auto retval                                     = InitResult< params, NB >{};
    auto& [diagonal, rhs]                           = retval;
    for (size_t p = 0; p != NP; ++p)
    {
        const auto jacobian = J[p];
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        const auto point         = SpaceTimePoint{points[p], time};
        const auto [fv, fd]      = fields.get(p);
        const auto kernel_result = kernel({fv, fd, point});
        const auto weight        = jacobian * quad_wgt[p];
        detail::precomputeDiagRhsImpl< params >(kernel_result,
                                                basis_vals->getMap().row(p),
                                                basis_ders.getPointMap(p),
                                                weight,
                                                diagonal,
                                                rhs,
                                                dir_inds,
                                                dir_vals);
    }
    return retval;
}

template < typename Kernel, KernelParams params, size_t NB, size_t NP >
auto precomputeOperatorDiagonalAndRhs(const BoundaryEquationKernel< Kernel, params >&                          kernel,
                                      const map::TabulatedBoundaryMapping< NB, NP, params.dimension >&         mapping,
                                      const map::FieldValuesAtPoints< NP, params.n_fields, params.dimension >& fields,
                                      std::span< const val_t, NP >                                             quad_wgt,
                                      val_t                                                                    time,
                                      DirichletInds< params, NB >                                              dir_inds,
                                      const DirichletVals< params, NB >& dir_vals) -> InitResult< params, NB >
{
    L3STER_PROFILE_FUNCTION;
    const auto& [J, basis_vals, basis_ders, points, normals] = mapping;
    auto retval                                              = InitResult< params, NB >{};
    auto& [diagonal, rhs]                                    = retval;
    for (size_t p = 0; p != NP; ++p)
    {
        const auto jacobian = J[p];
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        const auto point         = SpaceTimePoint{points[p], time};
        const auto [fv, fd]      = fields.get(p);
        const auto kernel_result = kernel({fv, fd, point, normals[p]});
        const auto weight        = jacobian * quad_wgt[p];
        detail::precomputeDiagRhsImpl< params >(kernel_result,
                                                basis_vals->getMap().row(p),
                                                basis_ders.getPointMap(p),
                                                weight,
                                                diagonal,
                                                rhs,
                                                dir_inds,
                                                dir_vals);
    }
    return retval;
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_EVALUATELOCALOPERATOR
