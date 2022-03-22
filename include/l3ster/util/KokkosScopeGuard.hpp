#ifndef L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
#define L3STER_UTIL_KOKKOSSCOPEGUARD_HPP

#include "Kokkos_Core.hpp"

struct KokkosScopeGuard
{
    KokkosScopeGuard(int& argc, char** argv) { Kokkos::initialize(argc, argv); }
    KokkosScopeGuard(const KokkosScopeGuard&) = delete;
    KokkosScopeGuard(KokkosScopeGuard&&)      = delete;
    KokkosScopeGuard& operator=(const KokkosScopeGuard&) = delete;
    KokkosScopeGuard& operator=(KokkosScopeGuard&&) = delete;
    ~KokkosScopeGuard() { Kokkos::finalize(); }
};
#endif // L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
