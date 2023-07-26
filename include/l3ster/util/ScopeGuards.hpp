#ifndef L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
#define L3STER_UTIL_KOKKOSSCOPEGUARD_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/SetStackSize.hpp"

namespace lstr
{
namespace util
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
    static constexpr auto required_mode            = MPI_THREAD_FUNNELED;
    constexpr auto        check_mpi_thread_support = [] {
        int        provided_mode{};
        const auto err_code = MPI_Query_thread(&provided_mode);
        lstr::detail::mpi::handleMPIError(err_code, "`MPI_Query_thread` failed");
        util::throwingAssert(provided_mode >= required_mode,
                             "The provided MPI installation appears not to have the required threading support: "
                                    "`MPI_THREAD_FUNNELED`\nIf you are initializing MPI yourself (i.e. not via L3STER scope "
                                    "guards), be sure to do so by calling `MPI_Init_thread` and passing `MPI_THREAD_FUNNELED` "
                                    "as the requirement, and *not* by calling `MPI_Init`");
    };
    constexpr auto check_initialized = []() -> bool {
        int        retval{};
        const auto err_code = MPI_Initialized(&retval);
        lstr::detail::mpi::handleMPIError(err_code, "`MPI_Initialized` failed");
        return retval;
    };
    constexpr auto check_not_finalized = [] {
        int        finalized{};
        const auto err_code = MPI_Finalized(&finalized);
        lstr::detail::mpi::handleMPIError(err_code, "`MPI_Finalized` failed");
        util::terminatingAssert(
            not finalized, "You are attempting to create `lstr::MpiScopeGuard` after `MPI_Finalize` has been called");
    };
    const auto initialize = [&] {
        int        dummy{};
        const auto err_code = MPI_Init_thread(&argc, &argv, required_mode, &dummy);
        lstr::detail::mpi::handleMPIError(err_code, "`MPI_Init_thread` failed");
    };

    check_not_finalized();
    if (not check_initialized())
    {
        initialize();
        m_is_owning = true;
    }
    check_mpi_thread_support();
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
} // namespace util

class L3sterScopeGuard
{
public:
    L3sterScopeGuard(int& argc, char** argv)
        : m_mpi_guard{argc, argv},
          m_kokkos_guard{argc, argv},
          m_stack_size_guard{util::detail::MaxStackSizeTracker::get()}
    {
        (void)util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized();
    }

private:
    util::MpiScopeGuard    m_mpi_guard;
    util::KokkosScopeGuard m_kokkos_guard;
    util::StackSizeGuard   m_stack_size_guard;
};

namespace util
{
class MaxParallelismGuard
{
public:
    MaxParallelismGuard(
        size_t max_threads = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNCores())
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
} // namespace util
} // namespace lstr
#endif // L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
