#ifndef L3STER_SIMSTRUCTURE_TIMEDEF_HPP
#define L3STER_SIMSTRUCTURE_TIMEDEF_HPP

#include "defs/Typedefs.h"

namespace lstr::def
{
struct Timeline
{
    constexpr Timeline(val_t t_begin_, val_t t_end_) : t_begin{t_begin_}, t_end{t_end_} {}

    val_t t_begin, t_end;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_TIMEDEF_HPP
