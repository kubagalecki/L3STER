#ifndef L3STER_UTIL_INCLUDEEIGEN_HPP
#define L3STER_UTIL_INCLUDEEIGEN_HPP

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_STACK_ALLOCATION_LIMIT 0 // NOLINT
#include "Eigen/Dense"

#include <concepts>

namespace lstr
{
namespace detail
{
template < typename T >
inline constexpr bool is_eigen_matrix_v = false;

template < typename P1, int P2, int P3, int P4, int P5, int P6 >
inline constexpr bool is_eigen_matrix_v< Eigen::Matrix< P1, P2, P3, P4, P5, P6 > > = true;

// Eigen requires that vectors be column major. If we want a row major matrix, which happens to also be a vector, we
// need to conditionally use the ColMajor flag.
int consteval eigenRowMajorIfNotVector(int size)
{
    return size > 1 ? Eigen::RowMajor : Eigen::ColMajor;
}
} // namespace detail

template < typename T >
concept EigenMatrix_c = detail::is_eigen_matrix_v< T >;

template < typename T >
concept EigenVector_c = EigenMatrix_c< T > and (T::ColsAtCompileTime == 1);

template < std::floating_point T, int size >
using EigenRowMajorSquareMatrix = Eigen::Matrix< T, size, size, detail::eigenRowMajorIfNotVector(size) >;

template < std::floating_point T, int rows, int cols >
using EigenRowMajorMatrix = Eigen::Matrix< T, rows, cols, detail::eigenRowMajorIfNotVector(cols) >;
} // namespace lstr
#endif // L3STER_UTIL_INCLUDEEIGEN_HPP
