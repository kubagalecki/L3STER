#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/BoundaryIntegralJacobian.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/math/IntegerMath.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/SetStackSize.hpp"

namespace lstr
{
namespace detail
{
template < typename T >
concept ValidKernelResult_c =
    Pair_c< T > and Array_c< typename T::first_type > and eigen::Matrix_c< typename T::first_type::value_type > and
    eigen::Vector_c< typename T::second_type > and
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
template < typename Kernel, dim_t dim, size_t n_fields >
inline constexpr std::size_t n_equations =
    kernel_result_t< Kernel, dim, n_fields >::first_type::value_type::RowsAtCompileTime;

template < int n_nodes, int n_fields >
auto computeFieldVals(const Eigen::Vector< val_t, n_nodes >&                   basis_vals,
                      const eigen::RowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
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
auto computeFieldDers(const eigen::RowMajorMatrix< val_t, dim, n_nodes >&      basis_ders,
                      const eigen::RowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< std::array< val_t, n_fields >, dim > retval;
    if constexpr (n_fields != 0)
    {
        const eigen::RowMajorMatrix< val_t, dim, n_fields > field_ders_packed = basis_ders * node_vals;
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
            std::copy_n(std::next(field_ders_packed.data(), dim_ind * n_fields), n_fields, retval[dim_ind].begin());
    }
    return retval;
}

template < size_t problem_size, size_t update_size >
class LocalSystemManager
{
    static constexpr size_t target_update_size = 128;
    static constexpr size_t updates_per_batch  = math::intDivRoundUp(target_update_size, update_size);
    static constexpr size_t batch_update_size  = update_size * updates_per_batch;
    using batch_update_matrix_t = Eigen::Matrix< val_t, problem_size, batch_update_size, Eigen::ColMajor >;

public:
    using matrix_t = eigen::RowMajorSquareMatrix< val_t, problem_size >;
    using vector_t = Eigen::Vector< val_t, problem_size >;
    using system_t = std::pair< matrix_t, vector_t >;

    static constexpr size_t required_stack_size = 2 * problem_size * batch_update_size * sizeof(val_t);

    inline void            setZero();
    inline const system_t& getSystem();

    template < int n_unknowns, int n_bases, int dim >
    void update(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
                const Eigen::Vector< val_t, update_size >&                                    kernel_rhs,
                const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                const eigen::RowMajorMatrix< val_t, dim, n_bases >&                           basis_ders,
                val_t                                                                         weight);

private:
    template < int n_unknowns, int n_bases, int dim >
    static auto
    makeBasisBlock(const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
                   const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                   const eigen::RowMajorMatrix< val_t, dim, n_bases >&                           basis_ders,
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

template < size_t problem_size, size_t update_size >
auto LocalSystemManager< problem_size, update_size >::tieBatchData(bool is_positive)
{
    return is_positive ? std::tie(*m_posw_buf, m_posw_batch_size) : std::tie(*m_negw_buf, m_negw_batch_size);
}

template < size_t problem_size, size_t update_size >
template < int n_unknowns, int n_bases, int dim >
auto LocalSystemManager< problem_size, update_size >::makeBasisBlock(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
    const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
    const eigen::RowMajorMatrix< val_t, dim, n_bases >&                           basis_ders,
    size_t                                                                        basis_ind)
{
    eigen::RowMajorMatrix< val_t, n_unknowns, update_size > retval =
        basis_vals[basis_ind] * kernel_result[0].transpose();
    for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
        retval += basis_ders(dim_ind, basis_ind) * kernel_result[dim_ind + 1].transpose();
    return retval;
}

template < size_t problem_size, size_t update_size >
template < int n_unknowns, int n_bases, int dim >
void LocalSystemManager< problem_size, update_size >::update(
    const std::array< Eigen::Matrix< val_t, update_size, n_unknowns >, dim + 1 >& kernel_result,
    const Eigen::Vector< val_t, update_size >&                                    kernel_rhs,
    const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
    const eigen::RowMajorMatrix< val_t, dim, n_bases >&                           basis_ders,
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
        m_system->second.template segment< n_unknowns >(row) += block * kernel_rhs * weight;
        batch_matrix.template block< n_unknowns, update_size >(row, col) = block * wgt_abs_sqrt;
    }
    if (++batch_size == updates_per_batch)
        flushFullBuf(batch_matrix, batch_size, is_wgt_positive ? 1. : -1.);
}

template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
auto& getLocalSystemManager()
{
    constexpr auto dim                = Element< ET, EO >::native_dim;
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns< Kernel, dim, n_fields >;
    constexpr auto update_size        = n_equations< Kernel, dim, n_fields >;
    auto&          retval             = util::getThreadLocal< LocalSystemManager< local_problem_size, update_size > >();
    retval.setZero();
    return retval;
}

template < size_t problem_size, size_t update_size >
const LocalSystemManager< problem_size, update_size >::system_t&
LocalSystemManager< problem_size, update_size >::getSystem()
{
    flush();
    m_system->first = m_system->first.template selfadjointView< Eigen::Lower >();
    return *m_system;
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::flush()
{
    flushBuf(*m_posw_buf, m_posw_batch_size, 1.);
    flushBuf(*m_negw_buf, m_negw_batch_size, -1.);
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::flushBuf(const batch_update_matrix_t& batch_matrix,
                                                               size_t&                      batch_size,
                                                               val_t                        weight)
{
    if (batch_size > 0)
        m_system->first.template selfadjointView< Eigen::Lower >().rankUpdate(
            batch_matrix.leftCols(batch_size * update_size), weight);
    batch_size = 0;
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::flushFullBuf(
    const LocalSystemManager::batch_update_matrix_t& batch_matrix, size_t& batch_size, val_t wgt)
{
    m_system->first.template selfadjointView< Eigen::Lower >().rankUpdate(batch_matrix, wgt);
    batch_size = 0;
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::setZero()
{
    m_system->first.setZero();
    m_system->second.setZero();
}
} // namespace detail

template < ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
const auto& assembleLocalSystem(auto&&                                                                      kernel,
                                const Element< ET, EO >&                                                    element,
                                const eigen::RowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                                const basis::ReferenceBasisAtQuadrature< ET, EO, QL >& basis_at_qps,
                                val_t                                                  time)
    requires detail::Kernel_c< decltype(kernel), Element< ET, EO >::native_dim, n_fields >
{
    L3STER_PROFILE_FUNCTION;
    const auto jacobi_mat_generator = map::getNatJacobiMatGenerator(element);
    auto&      local_system_manager = detail::getLocalSystemManager< decltype(kernel), ET, EO, n_fields >();
    const auto process_qp           = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat         = jacobi_mat_generator(point);
        const auto phys_basis_ders    = map::computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals         = detail::computeFieldVals(bas_vals, node_vals);
        const auto field_ders         = detail::computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords        = map::mapToPhysicalSpace(element, point);
        const auto [A, F]             = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time});
        const auto rank_update_weight = jacobi_mat.determinant() * weight;
        local_system_manager.update(A, F, bas_vals, phys_basis_ders, rank_update_weight);
    };
    for (size_t qp_ind = 0; qp_ind < basis_at_qps.quadrature.size; ++qp_ind)
        process_qp(basis_at_qps.quadrature.points[qp_ind],
                   basis_at_qps.quadrature.weights[qp_ind],
                   basis_at_qps.basis.values[qp_ind],
                   basis_at_qps.basis.derivatives[qp_ind]);
    return local_system_manager.getSystem();
}

template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
const auto&
assembleLocalBoundarySystem(Kernel&&                                                                    kernel,
                            BoundaryElementView< ET, EO >                                               el_view,
                            const eigen::RowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                            const basis::ReferenceBasisAtQuadrature< ET, EO, QL >&                      basis_at_qps,
                            val_t                                                                       time)
    requires detail::BoundaryKernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >
{
    L3STER_PROFILE_FUNCTION;
    const auto jacobi_mat_generator = map::getNatJacobiMatGenerator(*el_view);
    auto&      local_system_manager = detail::getLocalSystemManager< decltype(kernel), ET, EO, n_fields >();
    const auto process_qp           = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat      = jacobi_mat_generator(point);
        const auto phys_basis_ders = map::computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals      = detail::computeFieldVals(bas_vals, node_vals);
        const auto field_ders      = detail::computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords     = map::mapToPhysicalSpace(*el_view, point);
        const auto normal          = map::computeBoundaryNormal(el_view, jacobi_mat);
        const auto [A, F]    = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time}, normal);
        const auto bound_jac = map::computeBoundaryIntegralJacobian(el_view, jacobi_mat);
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
} // namespace lstr
#endif // L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
