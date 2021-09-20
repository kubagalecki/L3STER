#ifndef L3STER_COMM_MPICOMM_HPP
#define L3STER_COMM_MPICOMM_HPP

#include "util/Concepts.hpp"

#include "mpi.h"

#include <memory_resource>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lstr
{
namespace detail
{
template < typename T >
struct MpiType;
#define L3STER_MPI_TYPE_MAPPING_STRUCT(type, mpitype)                                                                  \
    template <>                                                                                                        \
    struct MpiType< type >                                                                                             \
    {                                                                                                                  \
        static MPI_Datatype value() { return mpitype; }                                                                \
    };
L3STER_MPI_TYPE_MAPPING_STRUCT(char, MPI_SIGNED_CHAR)                      // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(unsigned char, MPI_UNSIGNED_CHAR)           // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(short, MPI_SHORT)                           // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(unsigned short, MPI_UNSIGNED_SHORT)         // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(int, MPI_INT)                               // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(unsigned, MPI_UNSIGNED)                     // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(long, MPI_LONG)                             // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(unsigned long, MPI_UNSIGNED_LONG)           // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(long long, MPI_LONG_LONG)                   // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(unsigned long long, MPI_UNSIGNED_LONG_LONG) // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(float, MPI_FLOAT)                           // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(double, MPI_DOUBLE)                         // NOLINT
L3STER_MPI_TYPE_MAPPING_STRUCT(long double, MPI_LONG_DOUBLE)               // NOLINT
} // namespace detail

struct MpiScopeGuard
{
    inline MpiScopeGuard(int& argc, char**& argv);
    MpiScopeGuard(const MpiScopeGuard&) = delete;
    MpiScopeGuard(MpiScopeGuard&&)      = delete;
    MpiScopeGuard& operator=(const MpiScopeGuard&) = delete;
    MpiScopeGuard& operator=(MpiScopeGuard&&) = delete;
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
            "The current version of MPI does not support the required threading mode MPI_THREAD_SERIALIZED"};
}

class MpiComm
{
public:
    explicit MpiComm(MPI_Comm comm_ = MPI_COMM_WORLD) : comm{comm_} {}
    MpiComm(const MpiComm&) = delete;
    MpiComm(MpiComm&&)      = delete;
    MpiComm& operator=(const MpiComm&) = delete;
    MpiComm& operator=(MpiComm&&) = delete;
    ~MpiComm()                    = default;

    class Request
    {
    public:
        friend class MpiComm;

        Request(const Request&) = delete;
        Request& operator=(const Request&) = delete;
        inline Request(Request&&) noexcept;
        inline Request& operator=(Request&&) noexcept;
        ~Request() { wait(); }

        void        wait() { MPI_Wait(&request, MPI_STATUS_IGNORE); }
        inline bool test();

    private:
        Request() = default;

        MPI_Request request = MPI_REQUEST_NULL;
    };

    void abort() { MPI_Abort(comm, MPI_ERR_UNKNOWN); }

    [[nodiscard]] inline int getRank() const;
    [[nodiscard]] inline int getSize() const;

    // send
    template < arithmetic T >
    void send(const T* buf, size_t count, int dest, int tag = 0) const;
    template < arithmetic T >
    void send(const std::vector< T >& buf, int dest, int tag = 0) const
    {
        send(buf.data(), buf.size(), dest, tag);
    }
    template < arithmetic T >
    [[nodiscard]] Request sendAsync(const T* buf, size_t count, int dest, int tag = 0) const;
    template < arithmetic T >
    [[nodiscard]] Request sendAsync(const std::vector< T >& buf, int dest, int tag = 0) const
    {
        return sendAsync(buf.data(), buf.size(), dest, tag);
    }

    // recv
    template < arithmetic T >
    void receive(T* buf, size_t count, int source, int tag = 0) const;
    template < arithmetic T >
    T receive(int source, int tag = 0) const;
    template < arithmetic T >
    std::vector< T > receive(size_t count, int source, int tag = 0) const;
    template < arithmetic T >
    [[nodiscard]] Request receiveAsync(T* buf, size_t count, int source, int tag = 0) const;

    // reduce
    template < arithmetic T >
    void reduce(const T* send_buf, T* recv_buf, size_t count, int root, MPI_Op op) const;

    // observers
    [[nodiscard]] MPI_Comm& get() { return comm; }

private:
    MPI_Comm comm = MPI_COMM_WORLD;
};

MpiComm::Request::Request(MpiComm::Request&& other) noexcept : request(other.request)
{
    other.request = MPI_REQUEST_NULL;
}

MpiComm::Request& MpiComm::Request::operator=(MpiComm::Request&& other) noexcept
{
    wait();
    request       = other.request;
    other.request = MPI_REQUEST_NULL;
    return *this;
}

bool MpiComm::Request::test()
{
    int flag{};
    MPI_Test(&request, &flag, MPI_STATUS_IGNORE);
    return flag;
}

template < arithmetic T >
void MpiComm::send(const T* buf, size_t count, int dest, int tag) const
{
    if (MPI_Send(buf, count, detail::MpiType< T >::value(), dest, tag, comm))
        throw std::runtime_error{"MPI send failed"};
}

template < arithmetic T >
MpiComm::Request MpiComm::sendAsync(const T* buf, size_t count, int dest, int tag) const
{
    Request request;
    if (MPI_Isend(buf, count, detail::MpiType< T >::value(), dest, tag, comm, &request.request))
        throw std::runtime_error{"MPI asynchronous send failed"};
    return request;
}

template < arithmetic T >
void MpiComm::receive(T* buf, size_t count, int source, int tag) const
{
    if (MPI_Recv(buf, count, detail::MpiType< T >::value(), source, tag, comm, MPI_STATUS_IGNORE))
        throw std::runtime_error{"MPI receive failed"};
}

template < arithmetic T >
T MpiComm::receive(int source, int tag) const
{
    T ret_val{};
    receive(&ret_val, 1, source, tag);
    return ret_val;
}

template < arithmetic T >
std::vector< T > MpiComm::receive(size_t count, int source, int tag) const
{
    std::vector< T > ret_val(count);
    receive(ret_val.data(), count, source, tag);
    return ret_val;
}

template < arithmetic T >
MpiComm::Request MpiComm::receiveAsync(T* buf, size_t count, int source, int tag) const
{
    Request request;
    if (MPI_Irecv(buf, count, detail::MpiType< T >::value(), source, tag, comm, &request.request))
        throw std::runtime_error{"MPI asynchronous receive failed"};
    return request;
}

template < arithmetic T >
void MpiComm::reduce(const T* send_buf, T* recv_buf, size_t count, int root, MPI_Op op) const
{
    if (MPI_Reduce(send_buf, recv_buf, count, detail::MpiType< T >::value(), op, root, comm))
        throw std::runtime_error{"MPI reduce failed"};
}

int MpiComm::getRank() const
{
    int rank{};
    if (MPI_Comm_rank(comm, &rank))
        throw std::runtime_error{"MPI rank query failed"};
    return rank;
}

int MpiComm::getSize() const
{
    int size{};
    if (MPI_Comm_size(comm, &size))
        throw std::runtime_error{"MPI comm size query failed"};
    return size;
}
} // namespace lstr

#endif // L3STER_COMM_MPICOMM_HPP
