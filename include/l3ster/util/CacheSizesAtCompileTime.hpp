#ifndef L3STER_UTIL_CACHESIZESATCOMPILETIME_HPP
#define L3STER_UTIL_CACHESIZESATCOMPILETIME_HPP

#include <array>

namespace lstr::util
{
inline constexpr auto cache_sizes = std::array{
  32768uz,
  262144uz,
  2097152uz
};
}
#endif // L3STER_UTIL_CACHESIZESATCOMPILETIME_HPP
