#ifndef L3STER_POST_COMPUTENORM_HPP
#define L3STER_POST_COMPUTENORM_HPP

#include "l3ster/post/Integral.hpp"

namespace lstr
{
namespace detail
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
} // namespace detail

template < size_t n_fields = 0, AssemblyOptions opts = {}, el_o_t... orders >
auto computeNormL2(const MpiComm&                                       comm,
                   auto&&                                               eval_residual,
                   const mesh::MeshPartition< orders... >&              mesh,
                   mesh::detail::DomainIdRange_c auto&&                 domain_ids,
                   const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = {},
                   util::ConstexprValue< opts >                         options_ctwrpr   = {},
                   val_t                                                time             = 0.)
{
    const auto compute_squared_norm = [&eval_residual](const auto& vals, const auto& ders, const auto& point) noexcept
        requires requires {
            {
                std::invoke(eval_residual, vals, ders, point)
            } -> util::eigen::Vector_c;
        }
    {
        const auto residual = std::invoke(eval_residual, vals, ders, point);
        using ret_t         = std::remove_const_t< decltype(residual) >;
        return ret_t{residual.unaryExpr([](val_t v) { return v * v; })};
    };
    const util::eigen::Vector_c auto squared_norm = evalIntegral(comm,
                                                                 compute_squared_norm,
                                                                 mesh,
                                                                 std::forward< decltype(domain_ids) >(domain_ids),
                                                                 field_val_getter,
                                                                 detail::doubleQuadratureOrder(options_ctwrpr),
                                                                 time);
    using ret_t                                   = std::remove_const_t< decltype(squared_norm) >;
    return ret_t{squared_norm.cwiseSqrt()};
}

template < size_t n_fields = 0, AssemblyOptions opts = {}, el_o_t... orders >
auto computeBoundaryNormL2(const MpiComm&                                       comm,
                           auto&&                                               eval_residual,
                           const mesh::BoundaryView< orders... >&               boundary,
                           const SolutionManager::FieldValueGetter< n_fields >& field_val_getter = {},
                           util::ConstexprValue< opts >                         options_ctwrpr   = {},
                           val_t                                                time             = 0.)
{
    const auto compute_squared_norm = [&eval_residual](
        const auto& vals, const auto& ders, const auto& point, const auto& normal) noexcept
        requires requires {
            {
                std::invoke(eval_residual, vals, ders, point, normal)
            } -> util::eigen::Vector_c;
        }
    {
        const auto residual = std::invoke(eval_residual, vals, ders, point, normal);
        using ret_t         = std::remove_const_t< decltype(residual) >;
        return ret_t{residual.unaryExpr([](val_t v) { return v * v; })};
    };
    const util::eigen::Vector_c auto squared_norm = evalBoundaryIntegral(
        comm, compute_squared_norm, boundary, field_val_getter, detail::doubleQuadratureOrder(options_ctwrpr), time);
    using ret_t = std::remove_const_t< decltype(squared_norm) >;
    return ret_t{squared_norm.cwiseSqrt()};
}
} // namespace lstr
#endif // L3STER_POST_COMPUTENORM_HPP
