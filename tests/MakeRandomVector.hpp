#ifndef L3STER_MAKERANDOMVECTOR_HPP
#define L3STER_MAKERANDOMVECTOR_HPP

#include <algorithm>
#include <concepts>
#include <limits>
#include <random>
#include <vector>

namespace lstr
{
namespace detail
{
template < typename T >
requires std::floating_point< T > || std::integral< T >
struct rand_dist
{};

template < std::integral T >
struct rand_dist< T >
{
    using type = std::uniform_int_distribution< T >;
};

template < std::floating_point T >
struct rand_dist< T >
{
    using type = std::uniform_real_distribution< T >;
};
} // namespace detail

template < typename T >
requires std::floating_point< T > || std::integral< T >
using rand_dist_t = detail::rand_dist< T >::type;

template < typename T >
requires std::floating_point< T > || std::integral< T >
auto makeRandomVector(std::size_t size = 100,
                      T           min  = std::numeric_limits< T >::min(),
                      T           max  = std::numeric_limits< T >::max())
{
    thread_local std::mt19937 prng{std::random_device{}()};
    rand_dist_t< T >          random_dist{min, max};

    std::vector< T > result;
    result.reserve(size);
    std::generate_n(std::back_inserter(result), size, [&] { return random_dist(prng); });
    return result;
}
} // namespace lstr
#endif // L3STER_MAKERANDOMVECTOR_HPP
