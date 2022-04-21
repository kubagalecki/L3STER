#ifndef L3STER_MAPPING_COMPUTEJACOBIANSATQPOINTS_HPP
#define L3STER_MAPPING_COMPUTEJACOBIANSATQPOINTS_HPP

#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/quad/Quadrature.hpp"

namespace lstr
{
namespace detail
{
template < size_t native_dim >
using jacobian_t = Eigen::Matrix< val_t, native_dim, native_dim >;
} // namespace detail

template < ElementTypes ET, el_o_t EO, q_l_t QL, dim_t QD >
auto computeJacobiansAtQpoints(const Element< ET, EO >& element, const Quadrature< QL, QD >& quadrature)
{
    const auto                                                            jac_gen = getNatJacobiMatGenerator(element);
    std::array< detail::jacobian_t< Element< ET, EO >::native_dim >, QL > ret_val;
    std::ranges::transform(quadrature.getPoints(), begin(ret_val), [&](const auto& pc) { return jac_gen(Point{pc}); });
    return ret_val;
}
} // namespace lstr
#endif // L3STER_MAPPING_COMPUTEJACOBIANSATQPOINTS_HPP
