#ifndef L3STER_ALGSYS_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ALGSYS_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/common/KernelInterface.hpp"
#include "l3ster/common/Structs.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/math/IntegerMath.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"
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
        return eval_strategy != LocalEvalStrategy::LocalElement and (ET == Quad or ET == Hex);
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
template < int n_nodes, int n_fields >
auto computeFieldVals(const Eigen::Vector< val_t, n_nodes >&                         basis_vals,
                      const util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< val_t, n_fields > retval;
    if constexpr (n_fields != 0)
        Eigen::Map< Eigen::Vector< val_t, n_fields > >{retval.data()} = node_vals.transpose() * basis_vals;
    return retval;
}

template < int n_nodes, int n_fields, int dim >
auto computeFieldDers(const util::eigen::RowMajorMatrix< val_t, dim, n_nodes >&      basis_ders,
                      const util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< std::array< val_t, n_fields >, dim > retval;
    if constexpr (n_fields != 0)
    {
        const auto retval_data = reinterpret_cast< val_t* >(retval.data());
        Eigen::Map< util::eigen::RowMajorMatrix< val_t, dim, n_fields > >{retval_data} = basis_ders * node_vals;
    }
    return retval;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
class LocalSystemManager
{
    static constexpr size_t target_update_size = 16 * util::simd_width / sizeof(val_t);
    static constexpr size_t updates_per_batch  = math::intDivRoundUp(target_update_size, update_size);
    static constexpr size_t batch_update_size  = update_size * updates_per_batch;
    using batch_update_matrix_t = Eigen::Matrix< val_t, problem_size, batch_update_size, Eigen::ColMajor >;

public:
    LocalSystemManager() { util::requestStackSize< util::default_stack_size + required_stack_size >(); }

    using matrix_t = util::eigen::RowMajorSquareMatrix< val_t, problem_size >;
    using rhs_t    = Eigen::Matrix< val_t, problem_size, int{n_rhs} >;
    using system_t = std::pair< matrix_t, rhs_t >;

    static constexpr size_t required_stack_size = 2 * problem_size * batch_update_size * sizeof(val_t);

    inline auto setZero() -> LocalSystemManager&;
    inline auto getSystem() -> const system_t&;

    template < int n_unknowns, int n_bases, int dim >
    void update(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
                const Eigen::Matrix< val_t, update_size, int{n_rhs} >&                        kernel_rhs,
                const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                const util::eigen::RowMajorMatrix< val_t, dim, n_bases >&                     basis_ders,
                val_t                                                                         weight);

private:
    template < int n_unknowns, int n_bases, int dim >
    static auto
    makeBasisBlock(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
                   const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                   const util::eigen::RowMajorMatrix< val_t, dim, n_bases >&                     basis_ders,
                   size_t                                                                        basis_ind);

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
template < int n_unknowns, int n_bases, int dim >
auto LocalSystemManager< problem_size, update_size, n_rhs >::makeBasisBlock(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
    const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
    const util::eigen::RowMajorMatrix< val_t, dim, n_bases >&                     basis_ders,
    size_t                                                                        basis_ind)
{
    util::eigen::RowMajorMatrix< val_t, n_unknowns, update_size > retval =
        basis_vals[basis_ind] * kernel_result[0].transpose();
    for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
        retval += basis_ders(dim_ind, basis_ind) * kernel_result[dim_ind + 1].transpose();
    return retval;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
template < int n_unknowns, int n_bases, int dim >
void LocalSystemManager< problem_size, update_size, n_rhs >::update(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
    const Eigen::Matrix< val_t, update_size, int{n_rhs} >&                        kernel_rhs,
    const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
    const util::eigen::RowMajorMatrix< val_t, dim, n_bases >&                     basis_ders,
    val_t                                                                         weight)
{
    const bool is_wgt_positive      = weight >= 0.;
    auto [batch_matrix, batch_size] = tieBatchData(is_wgt_positive);
    const auto wgt_abs_sqrt         = std::sqrt(std::fabs(weight));
    for (size_t basis_ind = 0; basis_ind < static_cast< size_t >(n_bases); ++basis_ind)
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
void LocalSystemManager< problem_size, update_size, n_rhs >::flushFullBuf(
    const LocalSystemManager::batch_update_matrix_t& batch_matrix, size_t& batch_size, val_t wgt)
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

auto evalKernel(const auto& kernel,
                const auto& point,
                const auto& basis_vals,
                const auto& phys_basis_ders,
                const auto& node_vals,
                const auto& element_data,
                val_t       time,
                const auto&... args)
{
    const auto field_vals  = computeFieldVals(basis_vals, node_vals);
    const auto field_ders  = computeFieldDers(phys_basis_ders, node_vals);
    const auto phys_coords = map::mapToPhysicalSpace(element_data, point);
    const auto eval_point  = SpaceTimePoint{phys_coords, time};
    return kernel({field_vals, field_ders, eval_point, args...});
}

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
const auto& assembleLocalSystem(
    const DomainEquationKernel< Kernel, params >&                                                  kernel,
    const mesh::Element< ET, EO >&                                                                 element,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time)
{
    L3STER_PROFILE_FUNCTION;
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);
    const auto& el_data              = element.data;
    const auto  jacobi_gen           = map::getNatJacobiMatGenerator(el_data);
    auto&       local_system_manager = getLocalSystemManager< params, mesh::Element< ET, EO >::n_nodes >();
    const auto  process_qp = [&](const auto& point, val_t weight, const auto& basis_vals, const auto& ref_ders) {
        const auto [phys_ders, jacobian] = map::mapDomain< ET, EO >(jacobi_gen, point, ref_ders);
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        const auto [A, F]             = evalKernel(kernel, point, basis_vals, phys_ders, node_vals, el_data, time);
        const auto rank_update_weight = jacobian * weight;
        local_system_manager.update(A, F, basis_vals, phys_ders, rank_update_weight);
    };
    basis_at_qps.forEach(process_qp);
    return local_system_manager.getSystem();
}

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
const auto& assembleLocalSystem(
    const BoundaryEquationKernel< Kernel, params >&                                                kernel,
    mesh::BoundaryElementView< ET, EO >                                                            el_view,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time)
{
    L3STER_PROFILE_FUNCTION;
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);
    const auto& el_data              = el_view->data;
    const auto  jacobi_gen           = map::getNatJacobiMatGenerator(el_data);
    auto&       local_system_manager = getLocalSystemManager< params, mesh::Element< ET, EO >::n_nodes >();
    const auto  process_qp           = [&](auto point, val_t weight, const auto& basis_vals, const auto& ref_ders) {
        const auto side                          = el_view.getSide();
        const auto [phys_ders, jacobian, normal] = map::mapBoundary< ET, EO >(jacobi_gen, point, ref_ders, side);
        const auto [A, F] = evalKernel(kernel, point, basis_vals, phys_ders, node_vals, el_data, time, normal);
        const auto rank_update_weight = jacobian * weight;
        local_system_manager.update(A, F, basis_vals, phys_ders, rank_update_weight);
    };
    basis_at_qps.forEach(process_qp);
    return local_system_manager.getSystem();
}
} // namespace lstr::algsys
#endif // L3STER_ALGSYS_ASSEMBLELOCALSYSTEM_HPP
