#ifndef L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
#define L3STER_UTIL_KOKKOSSCOPEGUARD_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/defs/TrilinosTypedefs.h"

namespace lstr
{
namespace detail
{
struct MpiScopeGuard
{
    inline MpiScopeGuard(int& argc, char**& argv);
    MpiScopeGuard(const MpiScopeGuard&)                = delete;
    MpiScopeGuard& operator=(const MpiScopeGuard&)     = delete;
    MpiScopeGuard(MpiScopeGuard&&) noexcept            = default;
    MpiScopeGuard& operator=(MpiScopeGuard&&) noexcept = default;
    ~MpiScopeGuard() { MPI_Finalize(); }
};

MpiScopeGuard::MpiScopeGuard(int& argc, char**& argv)
{
    constexpr auto required_mode = MPI_THREAD_SERIALIZED;
    int            provided_mode{};
    int            mpi_status = MPI_Init_thread(&argc, &argv, required_mode, &provided_mode);
    if (mpi_status)
        throw std::runtime_error{"failed to initialize MPI"};
    if (provided_mode < required_mode)
        throw std::runtime_error{
            "The installed configuration of MPI does not support the required threading mode MPI_THREAD_SERIALIZED"};
}

struct KokkosScopeGuard
{
    KokkosScopeGuard() { Kokkos::initialize(); }
    KokkosScopeGuard(int& argc, char** argv) { Kokkos::initialize(argc, argv); }
    KokkosScopeGuard(const KokkosScopeGuard&)                = delete;
    KokkosScopeGuard& operator=(const KokkosScopeGuard&)     = delete;
    KokkosScopeGuard(KokkosScopeGuard&&) noexcept            = default;
    KokkosScopeGuard& operator=(KokkosScopeGuard&&) noexcept = default;
    ~KokkosScopeGuard() { Kokkos::finalize(); }
};
} // namespace detail

class L3sterScopeGuard
{
public:
    L3sterScopeGuard(int& argc, char** argv) : m_mpi_guard{argc, argv}, m_kokkos_guard{argc, argv} {}

private:
    detail::MpiScopeGuard    m_mpi_guard;
    detail::KokkosScopeGuard m_kokkos_guard;
};
} // namespace lstr
#endif // L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
