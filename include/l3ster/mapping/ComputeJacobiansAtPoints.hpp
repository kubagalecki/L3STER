#ifndef L3STER_MAPPING_COMPUTEJACOBIANSATPOINTS_HPP
#define L3STER_MAPPING_COMPUTEJACOBIANSATPOINTS_HPP

#include "l3ster/mapping/JacobiMat.hpp"
#include "l3ster/quad/Quadrature.hpp"

namespace lstr
{
namespace detail
{
template < size_t native_dim >
using jacobian_t = Eigen::Matrix< val_t, native_dim, native_dim >;
} // namespace detail

template < ElementTypes                                                  ET,
           el_o_t                                                        EO,
           std::convertible_to< Point< Element< ET, EO >::native_dim > > Point_t,
           size_t                                                        n_points >
auto computeJacobiansAtPoints(const Element< ET, EO >& element, const std::array< Point_t, n_points >& points)
{
    const auto jac_gen = getNatJacobiMatGenerator(element);
    std::array< detail::jacobian_t< Element< ET, EO >::native_dim >, n_points > retval;
    std::ranges::transform(points, begin(retval), [&](const auto& point) { return jac_gen(point); });
    return retval;
}
} // namespace lstr
#endif // L3STER_MAPPING_COMPUTEJACOBIANSATPOINTS_HPP
