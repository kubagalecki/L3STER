#ifndef L3STER_POST_COMPUTENORM_HPP
#define L3STER_POST_COMPUTENORM_HPP

#include "l3ster/post/Integral.hpp"

namespace lstr
{
template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, detail::FieldValGetter_c FvalGetter, detail::DomainIdRange_c R >
auto computeNormL2(const MpiComm&       comm,
                   auto&&               eval_residual,
                   const MeshPartition& mesh,
                   R&&                  domain_ids,
                   FvalGetter&&         field_val_getter,
                   val_t                time = 0.)
{
    const auto compute_squared_norm = [&eval_residual](const auto& vals, const auto& ders, const auto& point) noexcept
        requires requires {
                     {
                         std::invoke(eval_residual, vals, ders, point)
                         } -> EigenVector_c;
                 }
    {
        const auto residual = std::invoke(eval_residual, vals, ders, point);
        using ret_t         = std::remove_const_t< decltype(residual) >;
        return ret_t{residual.unaryExpr([](val_t v) { return v * v; })};
    };
    const EigenVector_c auto squared_norm = evalIntegral< BT, QT, QO >(comm,
                                                                       compute_squared_norm,
                                                                       mesh,
                                                                       std::forward< R >(domain_ids),
                                                                       std::forward< FvalGetter >(field_val_getter),
                                                                       time);
    using ret_t                           = std::remove_const_t< decltype(squared_norm) >;
    return ret_t{squared_norm.cwiseSqrt()};
}

template < BasisTypes BT, QuadratureTypes QT, q_o_t QO, detail::FieldValGetter_c FvalGetter >
auto computeBoundaryNormL2(const MpiComm&      comm,
                           auto&&              eval_residual,
                           const BoundaryView& boundary,
                           FvalGetter&&        field_val_getter,
                           val_t               time = 0.)
{
    const auto compute_squared_norm = [&eval_residual](
        const auto& vals, const auto& ders, const auto& point, const auto& normal) noexcept
        requires requires {
                     {
                         std::invoke(eval_residual, vals, ders, point, normal)
                         } -> EigenVector_c;
                 }
    {
        const auto residual = std::invoke(eval_residual, vals, ders, point, normal);
        using ret_t         = std::remove_const_t< decltype(residual) >;
        return ret_t{residual.unaryExpr([](val_t v) { return v * v; })};
    };
    const EigenVector_c auto squared_norm = evalBoundaryIntegral< BT, QT, QO >(
        comm, compute_squared_norm, boundary, std::forward< FvalGetter >(field_val_getter), time);
    using ret_t = std::remove_const_t< decltype(squared_norm) >;
    return ret_t{squared_norm.cwiseSqrt()};
}
} // namespace lstr
#endif // L3STER_POST_COMPUTENORM_HPP
