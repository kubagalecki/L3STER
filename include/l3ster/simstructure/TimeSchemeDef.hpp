#ifndef L3STER_SIMSTRUCTURE_TIMESCHEMEDEF_HPP
#define L3STER_SIMSTRUCTURE_TIMESCHEMEDEF_HPP

namespace lstr::def
{
enum struct TimeSchemes
{
    None,
    BDF2
};

template < TimeSchemes S >
struct TimeSolver
{};

template <>
struct TimeSolver< TimeSchemes::BDF2 >
{
    val_t dt = 0.;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_TIMESCHEMEDEF_HPP
