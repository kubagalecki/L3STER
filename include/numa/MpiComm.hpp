#ifndef L3STER_NUMA_MPICOMM_HPP
#define L3STER_NUMA_MPICOMM_HPP

#include "mpi.h"

#include <memory_resource>
#include <stdexcept>

namespace lstr
{
class MpiComm
{
public:
    inline MpiComm(const std::pmr::polymorphic_allocator<>& alloc_, int& argc, char**& argv);
    MpiComm(const MpiComm&) = delete;
    MpiComm(MpiComm&&)      = delete;
    MpiComm& operator=(const MpiComm&) = delete;
    MpiComm& operator=(MpiComm&&) = delete;
    ~MpiComm() { MPI_Finalize(); }

private:
    static inline void init(int& argc, char**& argv);

    std::pmr::polymorphic_allocator<> alloc;
};

MpiComm::MpiComm(const std::pmr::polymorphic_allocator<>& alloc_, int& argc, char**& argv) : alloc{alloc_}
{
    init(argc, argv);
}

void MpiComm::init(int& argc, char**& argv)
{
    constexpr auto required_mode = MPI_THREAD_SERIALIZED;
    int            provided_mode;
    int            mpi_status = MPI_Init_thread(&argc, &argv, required_mode, &provided_mode);
    if (mpi_status)
        throw std::runtime_error{"failed to initialize MPI"};
    if (provided_mode < required_mode)
        throw std::runtime_error{
            "The current version of MPI does not support the required threading mode MPI_THREAD_SERIALIZED"};
}
} // namespace lstr

#endif // L3STER_NUMA_MPICOMM_HPP
