#ifndef L3STER_ASSEMBLY_SPACETIMEPOINT_HPP
#define L3STER_ASSEMBLY_SPACETIMEPOINT_HPP

#include "../mesh/Point.hpp"

namespace lstr
{
struct SpaceTimePoint
{
    Point< 3 > space;
    val_t      time;
};
} // namespace lstr
#endif // L3STER_ASSEMBLY_SPACETIMEPOINT_HPP
