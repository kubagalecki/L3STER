#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/common/KernelInterface.hpp"
#include "l3ster/common/Structs.hpp"
#include "l3ster/mapping/BoundaryIntegralJacobian.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/math/IntegerMath.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/SetStackSize.hpp"

namespace lstr::glob_asm
{
template < int n_nodes, int n_fields >
auto computeFieldVals(const Eigen::Vector< val_t, n_nodes >&                         basis_vals,
                      const util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
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
auto computeFieldDers(const util::eigen::RowMajorMatrix< val_t, dim, n_nodes >&      basis_ders,
                      const util::eigen::RowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< std::array< val_t, n_fields >, dim > retval;
    if constexpr (n_fields != 0)
    {
        const util::eigen::RowMajorMatrix< val_t, dim, n_fields > field_ders_packed = basis_ders * node_vals;
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
            std::copy_n(std::next(field_ders_packed.data(), dim_ind * n_fields), n_fields, retval[dim_ind].begin());
    }
    return retval;
}

template < size_t problem_size, size_t update_size, size_t n_rhs >
class LocalSystemManager
{
    static constexpr size_t target_update_size = 128;
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

template < typename Kernel, KernelParams params, mesh::ElementType ET, el_o_t EO, q_l_t QL >
const auto& assembleLocalSystem(
    const DomainEquationKernel< Kernel, params >&                                                  kernel,
    const mesh::Element< ET, EO >&                                                                 element,
    const util::eigen::RowMajorMatrix< val_t, mesh::Element< ET, EO >::n_nodes, params.n_fields >& node_vals,
    const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                                         basis_at_qps,
    val_t                                                                                          time)
{
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);

    L3STER_PROFILE_FUNCTION;
    const auto jacobi_mat_generator = map::getNatJacobiMatGenerator(element);
    auto&      local_system_manager = getLocalSystemManager< params, mesh::Element< ET, EO >::n_nodes >();
    const auto process_qp           = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat      = jacobi_mat_generator(point);
        const auto phys_basis_ders = map::computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals      = computeFieldVals(bas_vals, node_vals);
        const auto field_ders      = computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords     = map::mapToPhysicalSpace(element, point);
        const auto eval_point      = SpaceTimePoint{phys_coords, time};
        const auto kernel_in = typename KernelInterface< params >::DomainInput{field_vals, field_ders, eval_point};
        const auto [A, F]    = kernel(kernel_in);
        const val_t jacobian = jacobi_mat.determinant();
        util::throwingAssert(jacobian > 0., "Encountered degenerate element ( |J| <= 0 )");
        local_system_manager.update(A, F, bas_vals, phys_basis_ders, jacobian * weight);
    };
    for (size_t qp_ind = 0; qp_ind < basis_at_qps.quadrature.size; ++qp_ind)
        process_qp(basis_at_qps.quadrature.points[qp_ind],
                   basis_at_qps.quadrature.weights[qp_ind],
                   basis_at_qps.basis.values[qp_ind],
                   basis_at_qps.basis.derivatives[qp_ind]);
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
    static_assert(params.dimension == mesh::ElementTraits< mesh::Element< ET, EO > >::native_dim);

    L3STER_PROFILE_FUNCTION;
    const auto jacobi_mat_generator = map::getNatJacobiMatGenerator(*el_view);
    auto&      local_system_manager = getLocalSystemManager< params, mesh::Element< ET, EO >::n_nodes >();
    const auto process_qp           = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat      = jacobi_mat_generator(point);
        const auto phys_basis_ders = map::computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals      = computeFieldVals(bas_vals, node_vals);
        const auto field_ders      = computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords     = map::mapToPhysicalSpace(*el_view, point);
        const auto eval_point      = SpaceTimePoint{phys_coords, time};
        const auto normal          = map::computeBoundaryNormal(el_view, jacobi_mat);
        const auto kernel_in =
            typename KernelInterface< params >::BoundaryInput{field_vals, field_ders, eval_point, normal};
        const auto [A, F]             = kernel(kernel_in);
        const auto bound_jac          = map::computeBoundaryIntegralJacobian(el_view, jacobi_mat);
        const auto rank_update_weight = bound_jac * weight;
        local_system_manager.update(A, F, bas_vals, phys_basis_ders, rank_update_weight);
    };
    for (size_t qp_ind = 0; qp_ind < basis_at_qps.quadrature.size; ++qp_ind)
        process_qp(basis_at_qps.quadrature.points[qp_ind],
                   basis_at_qps.quadrature.weights[qp_ind],
                   basis_at_qps.basis.values[qp_ind],
                   basis_at_qps.basis.derivatives[qp_ind]);
    return local_system_manager.getSystem();
}
} // namespace lstr::glob_asm
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
