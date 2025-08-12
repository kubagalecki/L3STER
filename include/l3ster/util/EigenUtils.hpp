#ifndef L3STER_UTIL_EIGENUTILS_HPP
#define L3STER_UTIL_EIGENUTILS_HPP

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_STACK_ALLOCATION_LIMIT 0 // NOLINT
#include "Eigen/Dense"

#include <concepts>

namespace lstr::util::eigen
{
namespace detail
{
template < typename T >
inline constexpr bool is_eigen_matrix_v = false;

template < typename P1, int P2, int P3, int P4, int P5, int P6 >
inline constexpr bool is_eigen_matrix_v< Eigen::Matrix< P1, P2, P3, P4, P5, P6 > > = true;

// Eigen requires that column-vectors be column major and row-vectors be row major. If we want a row major matrix, which
// happens to also be a vector, we need to conditionally use the ColMajor flag.
int consteval eigenRowMajorBit(int rows, int cols)
{
    if (rows == 1)
        return Eigen::RowMajor;
    else
        return cols == 1 ? Eigen::ColMajor : Eigen::RowMajor;
}

template < typename T >
inline constexpr bool is_statically_sized_martix = false;

template < typename P1, int P2, int P3, int P4, int P5, int P6 >
inline constexpr bool is_statically_sized_martix< Eigen::Matrix< P1, P2, P3, P4, P5, P6 > > =
    P2 != Eigen::Dynamic and P3 != Eigen::Dynamic;
} // namespace detail

template < typename T >
concept Matrix_c = detail::is_eigen_matrix_v< T >;

template < typename T >
concept Vector_c = Matrix_c< T > and (T::ColsAtCompileTime == 1);

template < std::floating_point T, int size >
using RowMajorSquareMatrix = Eigen::Matrix< T, size, size, detail::eigenRowMajorBit(size, size) >;

template < std::floating_point T, int rows, int cols >
using RowMajorMatrix = Eigen::Matrix< T, rows, cols, detail::eigenRowMajorBit(rows, cols) >;

template < std::floating_point T, int RCMaj = Eigen::RowMajor >
using DynamicallySizedMatrix =
    Eigen::Matrix< T, Eigen::Dynamic, Eigen::Dynamic, RCMaj, Eigen::Dynamic, Eigen::Dynamic >;

template < typename T >
concept StaticallySizedMatrix_c = Matrix_c< T > and detail::is_statically_sized_martix< T >;
} // namespace lstr::util::eigen
#endif // L3STER_UTIL_EIGENUTILS_HPP
