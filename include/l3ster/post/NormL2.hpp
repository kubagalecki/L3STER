#ifndef L3STER_POST_COMPUTENORM_HPP
#define L3STER_POST_COMPUTENORM_HPP

#include "l3ster/post/Integral.hpp"

namespace lstr
{
namespace post
{
template < AssemblyOptions opts >
consteval auto doubleQuadratureOrder(util::ConstexprValue< opts >)
{
    return util::ConstexprValue< std::invoke([] {
        auto retval = opts;
        retval.value_order *= 2;
        retval.derivative_order *= 2;
        return retval;
    }) >{};
}

auto getNormSquareComputer(const auto& kernel)
{
    return [&kernel](const auto& input, auto& out) {
        out = kernel(input);
        for (Eigen::Index i = 0; i != out.rows(); ++i)
            out[i] *= out[i];
    };
}
} // namespace post

template < AssemblyOptions opts = {}, typename Kernel, KernelParams params, el_o_t... orders >
auto computeNormL2(const MpiComm&                                              comm,
                   const ResidualDomainKernel< Kernel, params >&               eval_residual,
                   const mesh::MeshPartition< orders... >&                     mesh,
                   const util::ArrayOwner< d_id_t >&                           domain_ids,
                   const SolutionManager::FieldValueGetter< params.n_fields >& field_val_getter = {},
                   util::ConstexprValue< opts >                                options_ctwrpr   = {},
                   val_t time = 0.) -> KernelInterface< params >::Rhs
{
    const auto compute_squared_norm = post::getNormSquareComputer(eval_residual);
    const auto wrapped_sqn          = wrapDomainResidualKernel(compute_squared_norm, util::ConstexprValue< params >{});
    const auto squared_norm         = computeIntegral(
        comm, wrapped_sqn, mesh, domain_ids, field_val_getter, post::doubleQuadratureOrder(options_ctwrpr), time);
    return squared_norm.cwiseSqrt();
}

template < AssemblyOptions opts = {}, typename Kernel, KernelParams params, el_o_t... orders >
auto computeNormL2(const MpiComm&                                              comm,
                   const ResidualBoundaryKernel< Kernel, params >&             eval_residual,
                   const mesh::MeshPartition< orders... >&                     mesh,
                   const util::ArrayOwner< d_id_t >&                           boundary_ids,
                   const SolutionManager::FieldValueGetter< params.n_fields >& field_val_getter = {},
                   util::ConstexprValue< opts >                                options_ctwrpr   = {},
                   val_t time = 0.) -> KernelInterface< params >::Rhs
{
    const auto compute_squared_norm = post::getNormSquareComputer(eval_residual);
    const auto wrapped_sqn  = wrapBoundaryResidualKernel(compute_squared_norm, util::ConstexprValue< params >{});
    const auto squared_norm = computeIntegral(
        comm, wrapped_sqn, mesh, boundary_ids, field_val_getter, post::doubleQuadratureOrder(options_ctwrpr), time);
    return squared_norm.cwiseSqrt();
}
} // namespace lstr
#endif // L3STER_POST_COMPUTENORM_HPP
