#ifndef L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
#define L3STER_UTIL_KOKKOSSCOPEGUARD_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "Kokkos_Core.hpp"

namespace lstr
{
struct MpiScopeGuard
{
    inline MpiScopeGuard(int& argc, char**& argv);
    MpiScopeGuard(const MpiScopeGuard&)            = delete;
    MpiScopeGuard(MpiScopeGuard&&)                 = delete;
    MpiScopeGuard& operator=(const MpiScopeGuard&) = delete;
    MpiScopeGuard& operator=(MpiScopeGuard&&)      = delete;
    ~MpiScopeGuard() { MPI_Finalize(); }
};

struct KokkosScopeGuard
{
    KokkosScopeGuard(int& argc, char** argv) { Kokkos::initialize(argc, argv); }
    KokkosScopeGuard(const KokkosScopeGuard&)            = delete;
    KokkosScopeGuard(KokkosScopeGuard&&)                 = delete;
    KokkosScopeGuard& operator=(const KokkosScopeGuard&) = delete;
    KokkosScopeGuard& operator=(KokkosScopeGuard&&)      = delete;
    ~KokkosScopeGuard() { Kokkos::finalize(); }
};

struct L3sterScopeGuard
{
    inline L3sterScopeGuard(int& argc, char** argv);
    L3sterScopeGuard(const L3sterScopeGuard&)            = delete;
    L3sterScopeGuard(L3sterScopeGuard&&)                 = delete;
    L3sterScopeGuard& operator=(const L3sterScopeGuard&) = delete;
    L3sterScopeGuard& operator=(L3sterScopeGuard&&)      = delete;
    inline ~L3sterScopeGuard();
};

L3sterScopeGuard::L3sterScopeGuard(int& argc, char** argv)
{
    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    GlobalResource< KokkosScopeGuard >::initialize(argc, argv);
}

L3sterScopeGuard::~L3sterScopeGuard()
{
    GlobalResource< MpiScopeGuard >::finalize();
    GlobalResource< KokkosScopeGuard >::finalize();
}

MpiScopeGuard::MpiScopeGuard(int& argc, char**& argv)
{
    constexpr auto required_mode = MPI_THREAD_SERIALIZED;
    int            provided_mode{};
    int            mpi_status = MPI_Init_thread(&argc, &argv, required_mode, &provided_mode);
    if (mpi_status)
        throw std::runtime_error{"failed to initialize MPI"};
    if (provided_mode < required_mode)
        throw std::runtime_error{
            "The current version of MPI does not support the required threading mode MPI_THREAD_SERIALIZED"};
}
} // namespace lstr
#endif // L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
