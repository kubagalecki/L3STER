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
concept MpiBuf_c = std::ranges::contiguous_range< T > and std::ranges::sized_range< T > and
                   MpiType_c< std::ranges::range_value_t< T > >;
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
    class Request
    {
    public:
        friend class MpiComm;

        Request(const Request&)            = delete;
        Request& operator=(const Request&) = delete;
        inline Request(Request&&) noexcept;
        inline Request& operator=(Request&&) noexcept;
        ~Request() { waitIgnoreErr(); }

        void wait() { detail::handleMPIError(MPI_Wait(&m_request, MPI_STATUS_IGNORE), "MPI wait for request failed"); }
        void cancel() { detail::handleMPIError(MPI_Cancel(&m_request), "MPI async request cancellation failed"); }
        [[nodiscard]] inline bool test();

    private:
        Request() = default;
        void waitIgnoreErr() noexcept { MPI_Wait(&m_request, MPI_STATUS_IGNORE); }

        MPI_Request m_request = MPI_REQUEST_NULL;
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
        void closeIgnoreErr() { MPI_File_close(&m_file); }

        MPI_File m_file = MPI_FILE_NULL;
    };

    MpiComm(MPI_Comm comm = MPI_COMM_WORLD, bool is_owning = false) : m_comm{comm}, m_is_owning{is_owning} {}
    MpiComm(const MpiComm&)            = delete;
    MpiComm(MpiComm&&)                 = delete;
    MpiComm& operator=(const MpiComm&) = delete;
    MpiComm& operator=(MpiComm&&)      = delete;
    inline ~MpiComm();

    // send
    template < detail::MpiBuf_c T >
    void send(T&& send_range, int dest, int tag = 0) const;
    template < detail::MpiNonblockingBuf_c T >
    [[nodiscard]] Request sendAsync(T&& send_range, int dest, int tag = 0) const;

    // receive
    template < detail::MpiBuf_c R >
    void receive(R&& recv_range, int source, int tag = 0) const;
    template < detail::MpiType_c R >
    [[nodiscard]] R receive(int source, int tag = 0) const;
    template < detail::MpiType_c R >
    [[nodiscard]] std::vector< R > receive(size_t count, int source, int tag = 0) const;
    template < detail::MpiNonblockingBuf_c R >
    [[nodiscard]] Request receiveAsync(R&& recv_range, int source, int tag = 0) const;

    // collectives
    void barrier() const { MPI_Barrier(m_comm); }
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
    [[nodiscard]] MPI_Comm   get() const { return m_comm; }

    // topology-related
    [[nodiscard]] MpiComm distGraphCreate(ContiguousSizedRangeOf< int > auto&& sources,
                                          ContiguousSizedRangeOf< int > auto&& degrees,
                                          ContiguousSizedRangeOf< int > auto&& destinations,
                                          ContiguousSizedRangeOf< int > auto&& weights,
                                          bool                                 reorder) const;

    // misc
    inline FileHandle openFile(const char* file_name, int amode, MPI_Info info = MPI_INFO_NULL) const;
    void              abort() const { MPI_Abort(m_comm, MPI_ERR_UNKNOWN); }

private:
    MPI_Comm m_comm;
    bool     m_is_owning;
};

MpiComm::Request::Request(MpiComm::Request&& other) noexcept : m_request(other.m_request)
{
    other.m_request = MPI_REQUEST_NULL;
}
MpiComm::Request& MpiComm::Request::operator=(MpiComm::Request&& other) noexcept
{
    waitIgnoreErr();
    m_request = std::exchange(other.m_request, MPI_REQUEST_NULL);
    return *this;
}
bool MpiComm::Request::test()
{
    int flag{};
    detail::handleMPIError(MPI_Test(&m_request, &flag, MPI_STATUS_IGNORE), "MPI test async request failed");
    return flag;
}

MpiComm::FileHandle::FileHandle(MpiComm::FileHandle&& other) noexcept
    : m_file{std::exchange(other.m_file, MPI_FILE_NULL)}
{}
MpiComm::FileHandle& MpiComm::FileHandle::operator=(MpiComm::FileHandle&& other) noexcept
{
    closeIgnoreErr();
    m_file = std::exchange(other.m_file, MPI_FILE_NULL);
    return *this;
}
void MpiComm::FileHandle::preallocate(MPI_Offset size) const
{
    detail::handleMPIError(MPI_File_preallocate(m_file, size), "MPI file preallocate failed");
}
template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::FileHandle::readAtAsync(R&& read_range, MPI_Offset offset) const
{
    const auto       datatype = detail::MpiType< std::ranges::range_value_t< R > >::value();
    MpiComm::Request request;
    detail::handleMPIError(
        MPI_File_iread_at(
            m_file, offset, std::ranges::data(read_range), std::ranges::size(read_range), datatype, &request.m_request),
        "MPI asynchronous read from file at offset failed");
    return request;
}
template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::FileHandle::writeAtAsync(R&& write_range, MPI_Offset offset) const
{
    const auto       datatype = detail::MpiType< std::ranges::range_value_t< R > >::value();
    MpiComm::Request request;
    detail::handleMPIError(MPI_File_iwrite_at(m_file,
                                              offset,
                                              std::ranges::data(write_range),
                                              std::ranges::size(write_range),
                                              datatype,
                                              &request.m_request),
                           "MPI asynchronous write to file at offset failed");
    return request;
}
MpiComm::FileHandle MpiComm::openFile(const char* file_name, int amode, MPI_Info info) const
{
    FileHandle fh;
    detail::handleMPIError(MPI_File_open(m_comm, file_name, amode, info, &fh.m_file),
                           "MPI could not open the requested file");
    return fh;
}

template < detail::MpiBuf_c R >
void MpiComm::send(R&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(send_range);
    detail::handleMPIError(MPI_Send(buf_begin, buf_size, datatype, dest, tag, m_comm), "MPI send failed");
}

template < detail::MpiNonblockingBuf_c R >
MpiComm::Request MpiComm::sendAsync(R&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(send_range);
    Request request;
    detail::handleMPIError(MPI_Isend(buf_begin, buf_size, datatype, dest, tag, m_comm, &request.m_request),
                           "MPI asynchronous send failed");
    return request;
}

template < detail::MpiBuf_c R >
void MpiComm::receive(R&& recv_range, int source, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(recv_range);
    detail::handleMPIError(MPI_Recv(buf_begin, buf_size, datatype, source, tag, m_comm, MPI_STATUS_IGNORE),
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
    detail::handleMPIError(MPI_Irecv(buf_begin, buf_size, datatype, source, tag, m_comm, &request.m_request),
                           "MPI asynchronous receive failed");
    return request;
}

template < detail::MpiType_c T >
void MpiComm::reduce(const T* send_buf, T* recv_buf, size_t count, int root, MPI_Op op) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Reduce(send_buf, recv_buf, count, datatype, op, root, m_comm), "MPI reduce failed");
}

template < detail::MpiType_c T >
void MpiComm::allReduce(const T* send_buf, T* recv_buf, size_t count, MPI_Op op) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Allreduce(send_buf, recv_buf, count, datatype, op, m_comm), "MPI all-reduce failed");
}

template < detail::MpiType_c T >
void MpiComm::gather(const T* send_buf, T* recv_buf, size_t count, int root) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Gather(send_buf, count, datatype, recv_buf, count, datatype, root, m_comm),
                           "MPI gather failed");
}

template < detail::MpiType_c T >
void MpiComm::broadcast(T* message, int size, int root) const
{
    const auto datatype = detail::MpiType< T >::value();
    detail::handleMPIError(MPI_Bcast(message, size, datatype, root, m_comm), "MPI broadcast failed");
}

template < detail::MpiType_c T >
MpiComm::Request MpiComm::broadcastAsync(T* message, int size, int root) const
{
    const auto datatype = detail::MpiType< T >::value();
    Request    request{};
    detail::handleMPIError(MPI_Ibcast(message, size, datatype, root, m_comm, &request.m_request),
                           "MPI asynchronous broadcast failed");
    return request;
}

MpiComm MpiComm::distGraphCreate(ContiguousSizedRangeOf< int > auto&& sources,
                                 ContiguousSizedRangeOf< int > auto&& degrees,
                                 ContiguousSizedRangeOf< int > auto&& destinations,
                                 ContiguousSizedRangeOf< int > auto&& weights,
                                 bool                                 reorder) const
{
    MPI_Comm   graph_comm = MPI_COMM_NULL;
    const auto error_code = MPI_Dist_graph_create(m_comm,
                                                  static_cast< int >(std::ranges::ssize(sources)),
                                                  std::ranges::data(sources),
                                                  std::ranges::data(degrees),
                                                  std::ranges::data(destinations),
                                                  std::ranges::data(weights),
                                                  MPI_INFO_NULL,
                                                  reorder,
                                                  &graph_comm);
    detail::handleMPIError(error_code, "MPI_Dist_graph_create failed");
    return MpiComm{graph_comm, true};
}

template < detail::MpiType_c T >
MpiComm::Request MpiComm::exclusiveScanAsync(T* send_buf, T* recv_buf, int size, MPI_Op op) const
{
    const auto datatype = detail::MpiType< T >::value();
    Request    request{};
    detail::handleMPIError(MPI_Iexscan(send_buf, recv_buf, size, datatype, op, m_comm, &request.m_request),
                           "MPI asynchronous exclusive scan failed");
    return request;
}

int MpiComm::getRank() const
{
    int rank{};
    detail::handleMPIError(MPI_Comm_rank(m_comm, &rank), "MPI rank query failed");
    return rank;
}

int MpiComm::getSize() const
{
    int size{};
    detail::handleMPIError(MPI_Comm_size(m_comm, &size), "MPI m_comm size query failed");
    return size;
}

MpiComm::~MpiComm()
{
    if (m_is_owning)
        MPI_Comm_free(&m_comm);
}
} // namespace lstr
#endif // L3STER_COMM_MPICOMM_HPP
