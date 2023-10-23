#ifndef L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
#define L3STER_UTIL_KOKKOSSCOPEGUARD_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/util/GlobalResource.hpp"
#include "l3ster/util/HwlocWrapper.hpp"
#include "l3ster/util/SetStackSize.hpp"

#include <optional>

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
        int provided_mode{};
        L3STER_INVOKE_MPI(MPI_Query_thread, &provided_mode);
        util::throwingAssert(provided_mode >= required_mode,
                             "The provided MPI installation appears not to have the required threading support: "
                                    "`MPI_THREAD_FUNNELED`\nIf you are initializing MPI yourself (i.e. not via L3STER scope "
                                    "guards), be sure to do so by calling `MPI_Init_thread` and passing `MPI_THREAD_FUNNELED` "
                                    "as the requirement, and *not* by calling `MPI_Init`");
    };
    constexpr auto check_initialized = []() -> bool {
        int retval{};
        L3STER_INVOKE_MPI(MPI_Initialized, &retval);
        return retval;
    };
    constexpr auto check_not_finalized = [] {
        int finalized{};
        L3STER_INVOKE_MPI(MPI_Finalized, &finalized);
        util::terminatingAssert(
            not finalized, "You are attempting to create `lstr::MpiScopeGuard` after `MPI_Finalize` has been called");
    };
    const auto initialize = [&] {
        int dummy{};
        L3STER_INVOKE_MPI(MPI_Init_thread, &argc, &argv, required_mode, &dummy);
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
        : m_mpi_guard{argc, argv}, m_kokkos_guard{argc, argv}, m_stack_size_guard{util::MaxStackSizeTracker::get()}
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
#ifdef _OPENMP
    class OpenMPThreadGuard
    {
    public:
        explicit OpenMPThreadGuard(int num_threads) : m_num_threads_previous{omp_get_max_threads()}
        {
            omp_set_num_threads(num_threads);
        }
        OpenMPThreadGuard(const OpenMPThreadGuard&)            = delete;
        OpenMPThreadGuard& operator=(const OpenMPThreadGuard&) = delete;
        OpenMPThreadGuard(OpenMPThreadGuard&&)                 = default;
        OpenMPThreadGuard& operator=(OpenMPThreadGuard&&)      = default;
        ~OpenMPThreadGuard() { omp_set_num_threads(m_num_threads_previous); }

    private:
        int m_num_threads_previous;
    };
#else
    struct OpenMPThreadGuard
    {
        explicit OpenMPThreadGuard(int) {}
    };
#endif

    class ThreadGuards
    {
    public:
        explicit ThreadGuards(size_t threads)
            : m_tbb_guard{oneapi::tbb::global_control::max_allowed_parallelism, threads},
              m_openmp_guard{static_cast< int >(threads)}
        {}

    private:
        oneapi::tbb::global_control m_tbb_guard;
        OpenMPThreadGuard           m_openmp_guard;
    };

    static auto getCurrentMaxPar() -> size_t&
    {
        static size_t value = util::GlobalResource< util::hwloc::Topology >::getMaybeUninitialized().getNHwThreads();
        return value;
    }

public:
    explicit MaxParallelismGuard(size_t max_threads_unclamped) : m_previous{getCurrentMaxPar()}
    {
        const auto max_threads = std::clamp(max_threads_unclamped, size_t{1}, std::numeric_limits< size_t >::max());
        if (max_threads < m_previous)
        {
            m_thread_guards.emplace(max_threads);
            getCurrentMaxPar() = max_threads;
        }
    }

    MaxParallelismGuard(const MaxParallelismGuard&)            = delete;
    MaxParallelismGuard& operator=(const MaxParallelismGuard&) = delete;
    MaxParallelismGuard(MaxParallelismGuard&&)                 = default;
    MaxParallelismGuard& operator=(MaxParallelismGuard&&)      = default;
    ~MaxParallelismGuard()
    {
        if (m_thread_guards)
            getCurrentMaxPar() = m_previous;
    }

private:
    std::optional< ThreadGuards > m_thread_guards;
    size_t                        m_previous;
};
} // namespace util
} // namespace lstr
#endif // L3STER_UTIL_KOKKOSSCOPEGUARD_HPP
