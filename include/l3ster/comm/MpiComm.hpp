#ifndef L3STER_COMM_MPICOMM_HPP
#define L3STER_COMM_MPICOMM_HPP

#include "l3ster/util/Concepts.hpp"

extern "C"
{
#include "mpi.h"
}

#include <memory_resource>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace lstr
{
namespace detail
{
template < typename T >
struct MpiType
{};
#define L3STER_MPI_TYPE_MAPPING_STRUCT(type, mpitype)                                                                  \
    template <>                                                                                                        \
    struct MpiType< type >                                                                                             \
    {                                                                                                                  \
        static MPI_Datatype value()                                                                                    \
        {                                                                                                              \
            return mpitype;                                                                                            \
        }                                                                                                              \
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

inline void handleMPIError(int error, const char* message)
{
    if (error) [[unlikely]]
        throw std::runtime_error{message};
}

template < typename T >
concept MpiType_c = requires { MpiType< T >::value(); };
template < typename T >
concept MpiBuf_c = std::ranges::contiguous_range< T > and MpiType_c< std::ranges::range_value_t< T > >;
template < typename T >
concept MpiNonblockingBuf_c = MpiBuf_c< T > and std::ranges::borrowed_range< T >;

template < typename R >
auto decomposeMpiBuf(R&& buf)
{
    return std::make_tuple(MpiType< std::ranges::range_value_t< R > >::value(),
                           std::ranges::data(buf),
                           static_cast< int >(std::ranges::ssize(buf)));
}
} // namespace detail

class MpiComm
{
public:
    explicit MpiComm(MPI_Comm comm_ = MPI_COMM_WORLD) : comm{comm_} {}
    MpiComm(const MpiComm&)            = delete;
    MpiComm(MpiComm&&)                 = delete;
    MpiComm& operator=(const MpiComm&) = delete;
    MpiComm& operator=(MpiComm&&)      = delete;
    ~MpiComm()                         = default;

    class Request
    {
    public:
        friend class MpiComm;

        Request(const Request&)            = delete;
        Request& operator=(const Request&) = delete;
        inline Request(Request&&) noexcept;
        inline Request& operator=(Request&&) noexcept;
        ~Request() { waitIgnoreErr(); }

        void wait() { detail::handleMPIError(MPI_Wait(&request, MPI_STATUS_IGNORE), "MPI wait for request failed"); }
        void cancel() { detail::handleMPIError(MPI_Cancel(&request), "MPI async request cancellation failed"); }
        [[nodiscard]] inline bool test();

    private:
        Request() = default;
        void waitIgnoreErr() noexcept { MPI_Wait(&request, MPI_STATUS_IGNORE); }

        MPI_Request request = MPI_REQUEST_NULL;
    };

    class FileHandle
    {
    public:
        friend class MpiComm;

        FileHandle(const FileHandle&)            = delete;
        FileHandle& operator=(const FileHandle&) = delete;
        inline FileHandle(FileHandle&&) noexcept;
        inline FileHandle& operator=(FileHandle&&) noexcept;
        ~FileHandle() { closeIgnoreErr(); }

        inline void preallocate(MPI_Offset size) const;
        template < detail::MpiNonblockingBuf_c R >
        MpiComm::Request readAtAsync(R&& read_range, MPI_Offset offset) const;
        template < detail::MpiNonblockingBuf_c R >
        MpiComm::Request writeAtAsync(R&& write_range, MPI_Offset offset) const;

    private:
        FileHandle() = default;
        void closeIgnoreErr() { MPI_File_close(&file); }

        MPI_File file = MPI_FILE_NULL;
    };

    inline FileHandle openFile(const char* file_name, int amode, MPI_Info info = MPI_INFO_NULL) const;

    void abort() const { MPI_Abort(comm, MPI_ERR_UNKNOWN); }

    // send
    template < detail::MpiBuf_c T >
    void send(T&& send_range, int dest, int tag = 0) const;
    template < detail::MpiNonblockingBuf_c T >
    [[nodiscard]] Request sendAsync(T&& send_range, int dest, int tag = 0) const;

    // recv
    template < detail::MpiBuf_c R >
    void receive(R&& recv_range, int source, int tag = 0) const;
    template < detail::MpiType_c R >
    [[nodiscard]] R receive(int source, int tag = 0) const;
    template < detail::MpiType_c R >
    [[nodiscard]] std::vector< R > receive(size_t count, int source, int tag = 0) const;
    template < detail::MpiNonblockingBuf_c R >
    [[nodiscard]] Request receiveAsync(R&& recv_range, int source, int tag = 0) const;

    // collective comms
    void barrier() const { MPI_Barrier(comm); }
    template < detail::MpiType_c T >
    void reduce(const T* send_buf, T* recv_buf, size_t count, int root, MPI_Op op) const;
    template < detail::MpiType_c T >
    void allReduce(const T* send_buf, T* recv_buf, size_t count, MPI_Op op) const;
    template < detail::MpiType_c T >
    void gather(const T* send_buf, T* recv_buf, size_t count, int root) const;
    template < detail::MpiType_c T >
    void broadcast(T* message, int size, int root) const;
    template < detail::MpiType_c T >
    [[nodiscard]] Request broadcastAsync(T* message, int size, int root) const;
    template < detail::MpiType_c T >
    [[nodiscard]] Request exclusiveScanAsync(T* send_buf, T* recv_buf, int size, MPI_Op op) const;

    // observers
    [[nodiscard]] inline int getRank() const;
    [[nodiscard]] inline int getSize() const;
    [[nodiscard]] MPI_Comm   get() const { return comm; }

private:
    MPI_Comm comm = MPI_COMM_WORLD;
};

MpiComm::Request::Request(MpiComm::Request&& other) noexcept : request(other.request)
{
    other.request = MPI_REQUEST_NULL;
}
MpiComm::Request& MpiComm::Request::operator=(MpiComm::Request&& other) noexcept
{
    waitIgnoreErr();
    request = std::exchange(other.request, MPI_REQUEST_NULL);
    return *this;
}
bool MpiComm::Request::test()
{
    int flag{};
    detail::handleMPIError(MPI_Test(&request, &flag, MPI_STATUS_IGNORE), "MPI test async request failed");
    return flag;
}

MpiComm::FileHandle::FileHandle(MpiComm::FileHandle&& other) noexcept : file{other.file}
{
    other.file = MPI_FILE_NULL;
}
MpiComm::FileHandle& MpiComm::FileHandle::operator=(MpiComm::FileHandle&& other) noexcept
{
    closeIgnoreErr();
    file = std::exchange(other.file, MPI_FILE_NULL);
    return *this;
}
void MpiComm::FileHandle::preallocate(MPI_Offset size) const
{
    detail::handleMPIError(MPI_File_preallocate(file, size), "MPI file preallocate failed");
}
template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::FileHandle::readAtAsync(R&& read_range, MPI_Offset offset) const
{
    const auto       datatype = detail::MpiType< std::ranges::range_value_t< R > >::value();
    MpiComm::Request request;
    detail::handleMPIError(
        MPI_File_iread_at(
            file, offset, std::ranges::data(read_range), std::ranges::size(read_range), datatype, &request.request),
        "MPI asynchronous read from file at offset failed");
    return request;
}
template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::FileHandle::writeAtAsync(R&& write_range, MPI_Offset offset) const
{
    const auto       datatype = detail::MpiType< std::ranges::range_value_t< R > >::value();
    MpiComm::Request request;
    detail::handleMPIError(
        MPI_File_iwrite_at(
            file, offset, std::ranges::data(write_range), std::ranges::size(write_range), datatype, &request.request),
        "MPI asynchronous write to file at offset failed");
    return request;
}
MpiComm::FileHandle MpiComm::openFile(const char* file_name, int amode, MPI_Info info) const
{
    FileHandle fh;
    detail::handleMPIError(MPI_File_open(comm, file_name, amode, info, &fh.file),
                           "MPI could not open the requested file");
    return fh;
}

template < detail::MpiBuf_c R >
void MpiComm::send(R&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(send_range);
    detail::handleMPIError(MPI_Send(buf_begin, buf_size, datatype, dest, tag, comm), "MPI send failed");
}

template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::sendAsync(R&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(send_range);
    Request request;
    detail::handleMPIError(MPI_Isend(buf_begin, buf_size, datatype, dest, tag, comm, &request.request),
                           "MPI asynchronous send failed");
    return request;
}

template < detail::MpiBuf_c R >
void MpiComm::receive(R&& recv_range, int source, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(recv_range);
    detail::handleMPIError(MPI_Recv(buf_begin, buf_size, datatype, source, tag, comm, MPI_STATUS_IGNORE),
                           "MPI receive failed");
}

template < detail::MpiType_c T >
T MpiComm::receive(int source, int tag) const
{
    T retval{};
    receive(std::span{&retval, 1}, source, tag);
    return retval;
}

template < detail::MpiType_c T >
std::vector< T > MpiComm::receive(size_t count, int source, int tag) const
{
    std::vector< T > retval(count);
    receive(retval, source, tag);
    return retval;
}

template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::receiveAsync(R&& recv_range, int source, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(recv_range);
    Request request;
    detail::handleMPIError(MPI_Irecv(buf_begin, buf_size, datatype, source, tag, comm, &request.request),
                           "MPI asynchronous receive failed");
    return request;
}

template < detail::MpiType_c T >
void MpiComm::reduce(const T* send_buf, T* recv_buf, size_t count, int root, MPI_Op op) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Reduce(send_buf, recv_buf, count, datatype, op, root, comm), "MPI reduce failed");
}

template < detail::MpiType_c T >
void MpiComm::allReduce(const T* send_buf, T* recv_buf, size_t count, MPI_Op op) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Allreduce(send_buf, recv_buf, count, datatype, op, comm), "MPI all-reduce failed");
}

template < detail::MpiType_c T >
void MpiComm::gather(const T* send_buf, T* recv_buf, size_t count, int root) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Gather(send_buf, count, datatype, recv_buf, count, datatype, root, comm),
                           "MPI gather failed");
}

template < detail::MpiType_c T >
void MpiComm::broadcast(T* message, int size, int root) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Bcast(message, size, datatype, root, comm), "MPI broadcast failed");
}

template < detail::MpiType_c T >
MpiComm::Request MpiComm::broadcastAsync(T* message, int size, int root) const
{
    const auto datatype = detail::MpiType< T >::value();
    Request    request{};
    detail::handleMPIError(MPI_Ibcast(message, size, datatype, root, comm, &request.request),
                           "MPI asynchronous broadcast failed");
    return request;
}

template < detail::MpiType_c T >
MpiComm::Request MpiComm::exclusiveScanAsync(T* send_buf, T* recv_buf, int size, MPI_Op op) const
{
    const auto datatype = detail::MpiType< T >::value();
    Request    request{};
    detail::handleMPIError(MPI_Iexscan(send_buf, recv_buf, size, datatype, op, comm, &request.request),
                           "MPI asynchronous exclusive scan failed");
    return request;
}

int MpiComm::getRank() const
{
    int rank{};
    detail::handleMPIError(MPI_Comm_rank(comm, &rank), "MPI rank query failed");
    return rank;
}

int MpiComm::getSize() const
{
    int size{};
    detail::handleMPIError(MPI_Comm_size(comm, &size), "MPI comm size query failed");
    return size;
}
} // namespace lstr
#endif // L3STER_COMM_MPICOMM_HPP
