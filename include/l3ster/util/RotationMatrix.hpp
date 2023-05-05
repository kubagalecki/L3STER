#ifndef L3STER_UTIL_ROTATIONMATRIX_HPP
#define L3STER_UTIL_ROTATIONMATRIX_HPP

#include "l3ster/util/EigenUtils.hpp"

#include <cmath>
#include <concepts>

namespace lstr
{
template < auto maj = Eigen::ColMajor, std::floating_point T >
auto makeRotationMatrix3DX(T ang_rad) -> Eigen::Matrix< T, 3, 3, maj >
{
    const auto                    sin = std::sin(ang_rad);
    const auto                    cos = std::cos(ang_rad);
    Eigen::Matrix< T, 3, 3, maj > retval;
    retval(0, 0) = 1.;
    retval(1, 0) = 0.;
    retval(2, 0) = 0.;
    retval(0, 1) = 0.;
    retval(1, 1) = cos;
    retval(2, 1) = sin;
    retval(0, 2) = 0.;
    retval(1, 2) = -sin;
    retval(2, 2) = cos;
    return retval;
}

template < auto maj = Eigen::ColMajor, std::floating_point T >
auto makeRotationMatrix3DY(T ang_rad) -> Eigen::Matrix< T, 3, 3, maj >
{
    const auto                    sin = std::sin(ang_rad);
    const auto                    cos = std::cos(ang_rad);
    Eigen::Matrix< T, 3, 3, maj > retval;
    retval(0, 0) = cos;
    retval(1, 0) = 0.;
    retval(2, 0) = -sin;
    retval(0, 1) = 0.;
    retval(1, 1) = 1.;
    retval(2, 1) = 0.;
    retval(0, 2) = sin;
    retval(1, 2) = 0.;
    retval(2, 2) = cos;
    return retval;
}

template < auto maj = Eigen::ColMajor, std::floating_point T >
auto makeRotationMatrix3DZ(T ang_rad) -> Eigen::Matrix< T, 3, 3, maj >
{
    const auto                    sin = std::sin(ang_rad);
    const auto                    cos = std::cos(ang_rad);
    Eigen::Matrix< T, 3, 3, maj > retval;
    retval(0, 0) = cos;
    retval(1, 0) = sin;
    retval(2, 0) = 0.;
    retval(0, 1) = -sin;
    retval(1, 1) = cos;
    retval(2, 1) = 0.;
    retval(0, 2) = 0.;
    retval(1, 2) = 0.;
    retval(2, 2) = 1.;
    return retval;
}

template < auto maj = Eigen::ColMajor, std::floating_point T >
auto makeRotationMatrix2D(T ang_rad) -> Eigen::Matrix< T, 2, 2, maj >
{
    const auto                    sin = std::sin(ang_rad);
    const auto                    cos = std::cos(ang_rad);
    Eigen::Matrix< T, 2, 2, maj > retval;
    retval(0, 0) = cos;
    retval(1, 0) = -sin;
    retval(0, 1) = sin;
    retval(1, 1) = cos;
    return retval;
}
} // namespace lstr
#endif // L3STER_UTIL_ROTATIONMATRIX_HPP
