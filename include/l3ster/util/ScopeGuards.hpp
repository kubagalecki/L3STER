#ifndef L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
#define L3STER_UTIL_KOKKOSSCOPEGUARD_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/defs/TrilinosTypedefs.h"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/SetStackSize.hpp"

namespace lstr
{
namespace detail
{
class MpiScopeGuard
{
public:
    inline MpiScopeGuard(int& argc, char**& argv);
    MpiScopeGuard(const MpiScopeGuard&)            = delete;
    MpiScopeGuard& operator=(const MpiScopeGuard&) = delete;
    MpiScopeGuard(MpiScopeGuard&& other) noexcept : m_is_owning{std::exchange(other.m_is_owning, false)} {};
    MpiScopeGuard& operator=(MpiScopeGuard&& other) noexcept
    {
        if (this != &other)
            m_is_owning = std::exchange(other.m_is_owning, false);
        return *this;
    }
    ~MpiScopeGuard()
    {
        if (m_is_owning)
            MPI_Finalize();
    }

private:
    bool m_is_owning = false;
};

MpiScopeGuard::MpiScopeGuard(int& argc, char**& argv)
{
    int        initialized{};
    const auto init_check_err = MPI_Initialized(&initialized);
    detail::mpi::handleMPIError(init_check_err, "Failed to check MPI initialization status");
    if (initialized)
        return;

    constexpr auto required_mode = MPI_THREAD_FUNNELED;
    int            provided_mode{};
    int            mpi_status = MPI_Init_thread(&argc, &argv, required_mode, &provided_mode);
    detail::mpi::handleMPIError(mpi_status, "Failed to initialize MPI");
    m_is_owning = true;
    util::throwingAssert(
        provided_mode >= required_mode,
        "The installed configuration of MPI does not support the required threading mode MPI_THREAD_FUNNELED");
}

class KokkosScopeGuard
{
public:
    KokkosScopeGuard()
    {
        if (not Kokkos::is_initialized())
        {
            Kokkos::initialize();
            m_is_owning = true;
        }
    }
    KokkosScopeGuard(int& argc, char** argv)
    {
        if (not Kokkos::is_initialized())
        {
            Kokkos::initialize(argc, argv);
            m_is_owning = true;
        }
    }
    KokkosScopeGuard(const KokkosScopeGuard&)            = delete;
    KokkosScopeGuard& operator=(const KokkosScopeGuard&) = delete;
    KokkosScopeGuard(KokkosScopeGuard&& other) noexcept : m_is_owning{std::exchange(other.m_is_owning, false)} {};
    KokkosScopeGuard& operator=(KokkosScopeGuard&& other) noexcept
    {
        if (this != &other)
            m_is_owning = std::exchange(other.m_is_owning, false);
        return *this;
    }
    ~KokkosScopeGuard()
    {
        if (m_is_owning)
            Kokkos::finalize();
    }

private:
    bool m_is_owning;
};
} // namespace detail

class L3sterScopeGuard
{
public:
    L3sterScopeGuard(int& argc, char** argv)
        : m_mpi_guard{argc, argv},
          m_kokkos_guard{argc, argv},
          m_stack_size_guard{util::detail::MaxStackSizeTracker::get()}
    {
        (void)GlobalResource< util::hwloc::Topology >::getMaybeUninitialized();
    }

private:
    detail::MpiScopeGuard    m_mpi_guard;
    detail::KokkosScopeGuard m_kokkos_guard;
    util::StackSizeGuard     m_stack_size_guard;
};

namespace detail
{
class MaxParallelismGuard
{
public:
    MaxParallelismGuard(
        size_t max_threads = GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNCores())
        : m_max_threads{max_threads},
          m_tbb_control{oneapi::tbb::global_control::max_allowed_parallelism, max_threads}
#ifdef _OPENMP
          ,
          m_prev_omp_threads{omp_get_max_threads()}
#endif
    {
#ifdef _OPENMP
        omp_set_num_threads(static_cast< int >(m_max_threads));
#endif
    }

    MaxParallelismGuard(const MaxParallelismGuard&)            = delete;
    MaxParallelismGuard& operator=(const MaxParallelismGuard&) = delete;
    MaxParallelismGuard(MaxParallelismGuard&&)                 = default;
    MaxParallelismGuard& operator=(MaxParallelismGuard&&)      = default;
    ~MaxParallelismGuard()
    {
#ifdef _OPENMP
        omp_set_num_threads(m_prev_omp_threads);
#endif
    }

private:
    size_t                      m_max_threads;
    oneapi::tbb::global_control m_tbb_control;
#ifdef _OPENMP
    int m_prev_omp_threads;
#endif
};
} // namespace detail
} // namespace lstr
#endif // L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
