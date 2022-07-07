#ifndef L3STER_UTIL_INCLUDEEIGEN_HPP
#define L3STER_UTIL_INCLUDEEIGEN_HPP

#define EIGEN_DONT_PARALLELIZE
#define EIGEN_STACK_ALLOCATION_LIMIT 0
#include "Eigen/Dense"

namespace lstr
{
namespace detail
{
template < typename T >
inline constexpr bool is_eigen_matrix_v = false;

template < typename P1, int P2, int P3, int P4, int P5, int P6 >
inline constexpr bool is_eigen_matrix_v< Eigen::Matrix< P1, P2, P3, P4, P5, P6 > > = true;
} // namespace detail

template < typename T >
concept EigenMatrix_c = detail::is_eigen_matrix_v< T >;

template < typename T >
concept EigenVector_c = EigenMatrix_c< T > and (T::ColsAtCompileTime == 1);
} // namespace lstr
#endif // L3STER_UTIL_INCLUDEEIGEN_HPP
