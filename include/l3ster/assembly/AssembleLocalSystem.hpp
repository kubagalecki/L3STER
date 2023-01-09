#ifndef L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP
#define L3STER_ASSEMBLY_ASSEMBLELOCALSYSTEM_HPP

#include "l3ster/assembly/SpaceTimePoint.hpp"
#include "l3ster/basisfun/ReferenceBasisAtQuadrature.hpp"
#include "l3ster/mapping/BoundaryIntegralJacobian.hpp"
#include "l3ster/mapping/BoundaryNormal.hpp"
#include "l3ster/mapping/ComputePhysBasisDer.hpp"
#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/mesh/BoundaryElementView.hpp"

namespace lstr
{
namespace detail
{
template < typename T >
concept ValidKernelResult_c = Pair_c< T > and Array_c< typename T::first_type > and
                              EigenMatrix_c< typename T::first_type::value_type > and
                              EigenVector_c< typename T::second_type > and
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
auto computeFieldVals(const Eigen::Vector< val_t, n_nodes >&                 basis_vals,
                      const EigenRowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
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
auto computeFieldDers(const EigenRowMajorMatrix< val_t, dim, n_nodes >&      basis_ders,
                      const EigenRowMajorMatrix< val_t, n_nodes, n_fields >& node_vals)
{
    std::array< std::array< val_t, n_fields >, dim > retval;
    if constexpr (n_fields != 0)
    {
        const EigenRowMajorMatrix< val_t, dim, n_fields > field_ders_packed = basis_ders * node_vals;
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
            std::copy_n(std::next(field_ders_packed.data(), dim_ind * n_fields), n_fields, retval[dim_ind].begin());
    }
    return retval;
}

template < int n_equations, int n_unknowns, int n_bases, int dim >
auto makeRankUpdateMatrix(const std::array< Eigen::Matrix< val_t, n_equations, n_unknowns >, dim + 1 >& kernel_result,
                          const Eigen::Vector< val_t, n_bases >&                                        basis_vals,
                          const EigenRowMajorMatrix< val_t, dim, n_bases >&                             basis_ders)
{
    Eigen::Matrix< val_t, n_bases * n_unknowns, n_equations, Eigen::ColMajor > retval;
    for (size_t basis_ind = 0; basis_ind < static_cast< size_t >(n_bases); ++basis_ind)
    {
        auto block = retval.template block< n_unknowns, n_equations >(basis_ind * n_unknowns, 0);
        block      = basis_vals[basis_ind] * kernel_result[0].transpose();
        for (size_t dim_ind = 0; dim_ind < static_cast< size_t >(dim); ++dim_ind)
            block += basis_ders(dim_ind, basis_ind) * kernel_result[dim_ind + 1].transpose();
    }
    return retval;
}

template < size_t problem_size, size_t update_size >
class LocalSystemManager
{
public:
    using matrix_t        = EigenRowMajorSquareMatrix< val_t, problem_size >;
    using vector_t        = Eigen::Vector< val_t, problem_size >;
    using system_t        = std::pair< matrix_t, vector_t >;
    using system_ptr_t    = std::unique_ptr< system_t >;
    using update_matrix_t = Eigen::Matrix< val_t, problem_size, update_size, Eigen::ColMajor >;
    using update_vector_t = Eigen::Vector< val_t, update_size >;

    inline LocalSystemManager();
    inline system_ptr_t getSystem();
    inline void update(const update_matrix_t& update_matrix, const update_vector_t& update_vector, val_t update_weight);

private:
    static constexpr size_t target_update_size = 64;
    static constexpr size_t updates_per_batch =
        target_update_size % update_size == 0 ? target_update_size / update_size : target_update_size / update_size + 1;
    static constexpr size_t batch_update_size = update_size * updates_per_batch;
    using batch_update_matrix_t     = Eigen::Matrix< val_t, problem_size, batch_update_size, Eigen::ColMajor >;
    using batch_update_matrix_ptr_t = std::unique_ptr< batch_update_matrix_t >;

    inline void flush();
    inline void flushBuf(const batch_update_matrix_t& batch_matrix, size_t batch_size, val_t weight);
    inline void pushToBuf(const update_matrix_t& update_matrix, val_t update_weight);

    batch_update_matrix_ptr_t m_posw_buf = std::make_unique< batch_update_matrix_t >(),
                              m_negw_buf = std::make_unique< batch_update_matrix_t >();
    size_t       m_posw_batch_size{}, m_negw_batch_size{};
    system_ptr_t m_system = std::make_unique< system_t >();
};

template < typename Kernel, ElementTypes ET, el_o_t EO, size_t n_fields >
auto makeLocalSystemManager()
{
    constexpr auto dim                = Element< ET, EO >::native_dim;
    constexpr auto local_problem_size = Element< ET, EO >::n_nodes * n_unknowns< Kernel, dim, n_fields >;
    constexpr auto update_size        = n_equations< Kernel, dim, n_fields >;
    return LocalSystemManager< local_problem_size, update_size >{};
}

template < size_t problem_size, size_t update_size >
LocalSystemManager< problem_size, update_size >::LocalSystemManager()
{
    m_system->first.setZero();
    m_system->second.setZero();
}

template < size_t problem_size, size_t update_size >
LocalSystemManager< problem_size, update_size >::system_ptr_t
LocalSystemManager< problem_size, update_size >::getSystem()
{
    flush();
    m_system->first = m_system->first.template selfadjointView< Eigen::Lower >();
    return std::move(m_system);
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::update(const update_matrix_t& update_matrix,
                                                             const update_vector_t& update_vector,
                                                             val_t                  update_weight)
{
    m_system->second += update_matrix * update_vector * update_weight;
    pushToBuf(update_matrix, update_weight);
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::flush()
{
    flushBuf(*m_posw_buf, m_posw_batch_size, 1.);
    flushBuf(*m_negw_buf, m_negw_batch_size, -1.);
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::flushBuf(const batch_update_matrix_t& batch_matrix,
                                                               size_t                       batch_size,
                                                               val_t                        weight)
{
    if (batch_size > 0)
        m_system->first.template selfadjointView< Eigen::Lower >().rankUpdate(
            batch_matrix.leftCols(batch_size * update_size), weight);
}

template < size_t problem_size, size_t update_size >
void LocalSystemManager< problem_size, update_size >::pushToBuf(const update_matrix_t& update_matrix,
                                                                val_t                  update_weight)
{
    const auto  pos_wgt  = update_weight >= 0;
    const auto& buf      = pos_wgt ? m_posw_buf : m_negw_buf;
    auto&       buf_size = pos_wgt ? m_posw_batch_size : m_negw_batch_size;
    std::transform(std::execution::unseq,
                   update_matrix.data(),
                   std::next(update_matrix.data(), problem_size * update_size),
                   std::next(buf->data(), problem_size * update_size * buf_size++),
                   [weight_sqrt = std::sqrt(std::abs(update_weight))](val_t v) { return v * weight_sqrt; });
    if (buf_size == updates_per_batch)
    {
        m_system->first.template selfadjointView< Eigen::Lower >().rankUpdate(*buf, pos_wgt ? 1. : -1.);
        buf_size = 0;
    }
}
} // namespace detail

template < ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalSystem(auto&&                                                                    kernel,
                         const Element< ET, EO >&                                                  element,
                         const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                         const ReferenceBasisAtQuadrature< ET, EO, QL >&                           basis_at_qps,
                         val_t                                                                     time)
    requires detail::Kernel_c< decltype(kernel), Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(element);
    auto       local_system_manager = detail::makeLocalSystemManager< decltype(kernel), ET, EO, n_fields >();
    const auto process_qp           = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat         = jacobi_mat_generator(point);
        const auto phys_basis_ders    = computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals         = detail::computeFieldVals(bas_vals, node_vals);
        const auto field_ders         = detail::computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords        = mapToPhysicalSpace(element, point);
        const auto [A, F]             = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time});
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, bas_vals, phys_basis_ders);
        const auto rank_update_weight = jacobi_mat.determinant() * weight;
        local_system_manager.update(rank_update_matrix, F, rank_update_weight);
    };
    for (size_t qp_ind = 0; qp_ind < basis_at_qps.quadrature.size; ++qp_ind)
        process_qp(basis_at_qps.quadrature.points[qp_ind],
                   basis_at_qps.quadrature.weights[qp_ind],
                   basis_at_qps.basis.values[qp_ind],
                   basis_at_qps.basis.derivatives[qp_ind]);
    return local_system_manager.getSystem();
}

template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleLocalBoundarySystem(Kernel&&                                                                  kernel,
                                 BoundaryElementView< ET, EO >                                             el_view,
                                 const EigenRowMajorMatrix< val_t, Element< ET, EO >::n_nodes, n_fields >& node_vals,
                                 const ReferenceBasisAtQuadrature< ET, EO, QL >&                           basis_at_qps,
                                 val_t                                                                     time)
    requires detail::BoundaryKernel_c< Kernel, Element< ET, EO >::native_dim, n_fields >
{
    const auto jacobi_mat_generator = getNatJacobiMatGenerator(*el_view);
    auto       local_system_manager = detail::makeLocalSystemManager< decltype(kernel), ET, EO, n_fields >();
    const auto process_qp           = [&](auto point, val_t weight, const auto& bas_vals, const auto& ref_bas_ders) {
        const auto jacobi_mat      = jacobi_mat_generator(point);
        const auto phys_basis_ders = computePhysBasisDers(jacobi_mat, ref_bas_ders);
        const auto field_vals      = detail::computeFieldVals(bas_vals, node_vals);
        const auto field_ders      = detail::computeFieldDers(ref_bas_ders, node_vals);
        const auto phys_coords     = mapToPhysicalSpace(*el_view, point);
        const auto normal          = computeBoundaryNormal(el_view, jacobi_mat);
        const auto [A, F] = std::invoke(kernel, field_vals, field_ders, SpaceTimePoint{phys_coords, time}, normal);
        const auto rank_update_matrix = detail::makeRankUpdateMatrix(A, bas_vals, phys_basis_ders);
        const auto bound_jac          = computeBoundaryIntegralJacobian(el_view, jacobi_mat);
        const auto rank_update_weight = bound_jac * weight;
        local_system_manager.update(rank_update_matrix, F, rank_update_weight);
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
