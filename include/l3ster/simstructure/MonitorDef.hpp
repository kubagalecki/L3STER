#ifndef L3STER_SIMSTRUCTURE_MONITORDEF_HPP
#define L3STER_SIMSTRUCTURE_MONITORDEF_HPP

namespace lstr::def
{
enum struct MonitorTypes
{
    Integral,
    Flux,
    BoundaryNormalIntegral
};

template < MonitorTypes MT >
struct MonitorParams;

template < MonitorTypes MT >
struct Monitor
{
    MonitorParams< MT > params;
};

template < MonitorTypes... MT >
struct MonitorSet
{
    constexpr MonitorSet(const Monitor< MT >&... monitors_) : monitors{monitors...} {}

    std::tuple< Monitor< MT >... > monitors;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_MONITORDEF_HPP
