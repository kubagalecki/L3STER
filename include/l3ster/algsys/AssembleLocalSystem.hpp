#ifndef L3STER_ALGSYS_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ALGSYS_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/common/KernelInterface.hpp"
#include "l3ster/common/Structs.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/math/IntegerMath.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/SetStackSize.hpp"
#include "l3ster/util/Simd.hpp"

namespace lstr
{
enum struct LocalEvalStrategy
{
    Auto,
    LocalElement,
    SumFactorization,
    SumFactorizationOddEvenDecomposition
};

struct AssemblyOptions
{
    q_o_t                value_order      = 1;
    q_o_t                derivative_order = 0;
    basis::BasisType     basis_type       = basis::BasisType::Lagrange;
    quad::QuadratureType quad_type        = quad::QuadratureType::GaussLegendre;
    LocalEvalStrategy    eval_strategy    = LocalEvalStrategy::Auto;

    [[nodiscard]] constexpr q_o_t order(el_o_t elem_order) const
    {
        return static_cast< q_o_t >(value_order * elem_order + derivative_order * (elem_order - 1));
    }
    [[nodiscard]] constexpr bool useSumFactorization(mesh::ElementType             ET,
                                                     [[maybe_unused]] el_o_t       EO,
                                                     [[maybe_unused]] KernelParams kernel_params) const
    {
        using enum mesh::ElementType;
        return eval_strategy != LocalEvalStrategy::LocalElement and
               std::ranges::contains(std::array{Quad, Quad2, Hex, Hex2}, ET);
    }
    [[nodiscard]] constexpr bool useOddEven(el_o_t EO) const
    {
        if (eval_strategy == LocalEvalStrategy::Auto)
            return EO >= 2 && EO <= 6;
        return eval_strategy != LocalEvalStrategy::SumFactorization;
    }
};
} // namespace lstr

namespace lstr::algsys
{
template < size_t problem_size, size_t update_size, size_t n_rhs >
class LocalSystemManager
{
    static constexpr size_t target_update_size = 16 * util::simd_width / sizeof(val_t);
    static constexpr size_t updates_per_batch  = math::intDivRoundUp(target_update_size, update_size);
    static constexpr size_t batch_update_size  = update_size * updates_per_batch;
    using batch_update_matrix_t = Eigen::Matrix< val_t, problem_size, batch_update_size, Eigen::ColMajor >;

public:
    LocalSystemManager() { util::requestStackSize< required_stack_size >(); }

    using matrix_t = util::eigen::RowMajorSquareMatrix< val_t, problem_size >;
    using rhs_t    = Eigen::Matrix< val_t, problem_size, int{n_rhs} >;
    using system_t = std::pair< matrix_t, rhs_t >;

    static constexpr size_t required_stack_size = 2 * problem_size * batch_update_size * sizeof(val_t);

    inline auto setZero() -> LocalSystemManager&;
    inline auto getSystem() -> const system_t&;

    template < int n_unknowns, size_t dimp1, typename Vals, typename Ders >
    void update(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
                const Eigen::Matrix< val_t, update_size, int{n_rhs} >&                      kernel_rhs,
                const Vals&                                                                 basis_vals,
                const Ders&                                                                 basis_ders,
                val_t                                                                       weight);

private:
    template < int n_unknowns, size_t dimp1, typename Vals, typename Ders >
    static auto
    makeBasisBlock(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
                   const Vals&                                                                 basis_vals,
                   const Ders&                                                                 basis_ders,
                   size_t                                                                      basis_ind);

    inline auto tieBatchData(bool is_positive);
    inline void flush();
    inline void flushBuf(const batch_update_matrix_t& batch_matrix, size_t& batch_size, val_t weight);
    inline void flushFullBuf(const batch_update_matrix_t& batch_matrix, size_t& batch_size, val_t wgt);

    std::unique_ptr< batch_update_matrix_t > m_posw_buf = std::make_unique< batch_update_matrix_t >(),
                                             m_negw_buf = std::make_unique< batch_update_matrix_t >();
    size_t                      m_posw_batch_size{}, m_negw_batch_size{};
    std::unique_ptr< system_t > m_system = std::make_unique< system_t >();
};

template < size_t problem_size, size_t update_size, size_t n_rhs >
auto LocalSystemManager< problem_size, update_size, n_rhs >::tieBatchData(bool is_positive)
{
    return is_positive ? std::tie(*m_posw_buf, m_posw_batch_size) : std::tie(*m_negw_buf, m_negw_batch_size);
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
template < int n_unknowns, size_t dimp1, typename Vals, typename Ders >
auto LocalSystemManager< problem_size, update_size, n_rhs >::makeBasisBlock(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
    const Vals&                                                                 basis_vals,
    const Ders&                                                                 basis_ders,
    size_t                                                                      basis_ind)
{
    constexpr size_t                                              dim = dimp1 - 1;
    util::eigen::RowMajorMatrix< val_t, n_unknowns, update_size > retval =
        basis_vals[basis_ind] * kernel_result[0].transpose();
    for (size_t dim_ind = 0; dim_ind < dim; ++dim_ind)
        retval += basis_ders(basis_ind, dim_ind) * kernel_result[dim_ind + 1].transpose();
    return retval;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
template < int n_unknowns, size_t dimp1, typename Vals, typename Ders >
void LocalSystemManager< problem_size, update_size, n_rhs >::update(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dimp1 >& kernel_result,
    const Eigen::Matrix< val_t, update_size, int{n_rhs} >&                      kernel_rhs,
    const Vals&                                                                 basis_vals,
    const Ders&                                                                 basis_ders,
    val_t                                                                       weight)
{
    const bool is_wgt_positive      = weight >= 0.;
    auto [batch_matrix, batch_size] = tieBatchData(is_wgt_positive);
    const auto wgt_abs_sqrt         = std::sqrt(std::fabs(weight));
    for (size_t basis_ind = 0; basis_ind < static_cast< size_t >(basis_vals.size()); ++basis_ind)
    {
        const auto block = makeBasisBlock(kernel_result, basis_vals, basis_ders, basis_ind);
        const auto row   = basis_ind * n_unknowns;
        const auto col   = batch_size * update_size;
        m_system->second.template block< n_unknowns, int{n_rhs} >(row, 0) += block * kernel_rhs * weight;
        batch_matrix.template block< n_unknowns, update_size >(row, col) = block * wgt_abs_sqrt;
    }
    if (++batch_size == updates_per_batch)
        flushFullBuf(batch_matrix, batch_size, is_wgt_positive ? 1. : -1.);
}

template < KernelParams params, size_t n_nodes >
auto& getLocalSystemManager()
{
    constexpr auto local_problem_size = n_nodes * params.n_unknowns;
    using local_system_t              = LocalSystemManager< local_problem_size, params.n_equations, params.n_rhs >;
    return util::getThreadLocal< local_system_t >().setZero();
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
auto LocalSystemManager< problem_size, update_size, n_rhs >::getSystem() -> const system_t&
{
    flush();
    m_system->first = m_system->first.template selfadjointView< Eigen::Lower >();
    return *m_system;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
void LocalSystemManager< problem_size, update_size, n_rhs >::flush()
{
    flushBuf(*m_posw_buf, m_posw_batch_size, 1.);
    flushBuf(*m_negw_buf, m_negw_batch_size, -1.);
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
void LocalSystemManager< problem_size, update_size, n_rhs >::flushBuf(const batch_update_matrix_t& batch_matrix,
                                                                      size_t&                      batch_size,
                                                                      val_t                        weight)
{
    if (batch_size > 0)
        m_system->first.template selfadjointView< Eigen::Lower >().rankUpdate(
            batch_matrix.leftCols(batch_size * update_size), weight);
    batch_size = 0;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
void LocalSystemManager< problem_size, update_size, n_rhs >::flushFullBuf(const batch_update_matrix_t& batch_matrix,
                                                                          size_t&                      batch_size,
                                                                          val_t                        wgt)
{
    m_system->first.template selfadjointView< Eigen::Lower >().rankUpdate(batch_matrix, wgt);
    batch_size = 0;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
auto LocalSystemManager< problem_size, update_size, n_rhs >::setZero() -> LocalSystemManager&
{
    m_system->first.setZero();
    m_system->second.setZero();
    return *this;
}

template < typename Kernel, KernelParams params, size_t NB, size_t NP >
const auto& assembleLocalSystem(const DomainEquationKernel< Kernel, params >&                            kernel,
                                const map::TabulatedDomainMapping< NB, NP, params.dimension >&           mapping,
                                const map::FieldValuesAtPoints< NP, params.n_fields, params.dimension >& fields,
                                std::span< const val_t, NP >                                             quad_weights,
                                val_t                                                                    time)
{
    L3STER_PROFILE_FUNCTION;
    const auto& [J, basis_vals, basis_ders, points] = mapping;
    auto& local_system_manager                      = getLocalSystemManager< params, NB >();
    for (size_t p = 0; p != NP; ++p)
    {
        const auto jacobian = J[p];
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        const auto point    = SpaceTimePoint{points[p], time};
        const auto [fv, fd] = fields.get(p);
        const auto [A, F]   = kernel({fv, fd, point});
        const auto weight   = jacobian * quad_weights[p];
        local_system_manager.update(A, F, basis_vals->getMap().row(p), basis_ders.getPointMap(p), weight);
    }
    return local_system_manager.getSystem();
}

template < typename Kernel, KernelParams params, size_t NB, size_t NP >
const auto& assembleLocalSystem(const BoundaryEquationKernel< Kernel, params >&                          kernel,
                                const map::TabulatedBoundaryMapping< NB, NP, params.dimension >&         mapping,
                                const map::FieldValuesAtPoints< NP, params.n_fields, params.dimension >& fields,
                                std::span< const val_t, NP >                                             quad_weights,
                                val_t                                                                    time)
{
    L3STER_PROFILE_FUNCTION;
    const auto& [J, basis_vals, basis_ders, points, normals] = mapping;
    auto& local_system_manager                               = getLocalSystemManager< params, NB >();
    for (size_t p = 0; p != NP; ++p)
    {
        const auto jacobian = J[p];
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        const auto point    = SpaceTimePoint{points[p], time};
        const auto [fv, fd] = fields.get(p);
        const auto [A, F]   = kernel({fv, fd, point, normals[p]});
        const auto weight   = jacobian * quad_weights[p];
        local_system_manager.update(A, F, basis_vals->getMap().row(p), basis_ders.getPointMap(p), weight);
    }
    return local_system_manager.getSystem();
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_ASSEMBLELOCALSYSTEM_HPP
