#ifndef L3STER_MATH_INTEGERMATH_HPP
#define L3STER_MATH_INTEGERMATH_HPP

#include <concepts>

namespace lstr::math
{
template < std::integral T >
constexpr T intDivRoundUp(T enumerator, T denominator)
{
    const auto rem  = enumerator % denominator;
    const auto quot = enumerator / denominator;
    return rem == 0 ? quot : quot + 1;
}
} // namespace lstr::math
#endif // L3STER_MATH_INTEGERMATH_HPP
