#ifndef L3STER_COMM_MPICOMM_HPP
#define L3STER_COMM_MPICOMM_HPP

#include "l3ster/util/Concepts.hpp"

extern "C"
{
#include "mpi.h"
}

#include <iterator>
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
concept MpiBorrowedBuf_c = MpiBuf_c< T > and std::ranges::borrowed_range< T >;

auto decomposeMpiBuf(MpiBuf_c auto&& buf)
{
    return std::make_tuple(MpiType< std::ranges::range_value_t< decltype(buf) > >::value(),
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
        void cancel() { detail::handleMPIError(MPI_Cancel(&m_request), "MPI request cancellation failed"); }
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
        auto        readAtAsync(detail::MpiBorrowedBuf_c auto&& data, MPI_Offset offset) const -> MpiComm::Request;
        auto        writeAtAsync(detail::MpiBorrowedBuf_c auto&& data, MPI_Offset offset) const -> MpiComm::Request;

    private:
        FileHandle() = default;
        void closeIgnoreErr() { MPI_File_close(&m_file); }

        MPI_File m_file = MPI_FILE_NULL;
    };

    inline MpiComm(MPI_Comm comm);
    MpiComm(const MpiComm&)            = delete;
    MpiComm& operator=(const MpiComm&) = delete;
    MpiComm(MpiComm&& other) noexcept : m_comm{std::exchange(other.m_comm, MPI_COMM_NULL)} {}
    inline MpiComm& operator=(MpiComm&& other) noexcept;
    ~MpiComm() { freeOwnedComm(); }

    // send
    void               send(detail::MpiBuf_c auto&& send_range, int dest, int tag = 0) const;
    [[nodiscard]] auto sendAsync(detail::MpiBorrowedBuf_c auto&& data, int dest, int tag = 0) const -> Request;

    // receive
    void               receive(detail::MpiBuf_c auto&& recv_range, int src, int tag = 0) const;
    [[nodiscard]] auto receiveAsync(detail::MpiBorrowedBuf_c auto&& data, int src, int tag = 0) const -> Request;

    // collectives
    void barrier() const { detail::handleMPIError(MPI_Barrier(m_comm), "MPI_Barrier failed"); }
    void reduce(detail::MpiBuf_c auto&& data, auto out_it, int root, MPI_Op op) const
        requires std::output_iterator< decltype(out_it), std::ranges::range_value_t< decltype(data) > > and
                 std::contiguous_iterator< decltype(out_it) >
    {
        const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
        detail::handleMPIError(MPI_Reduce(buf_begin, std::addressof(*out_it), buf_size, datatype, op, root, m_comm),
                               "MPI reduce failed");
    }
    void allReduce(detail::MpiBuf_c auto&& data, auto out_it, MPI_Op op) const
        requires std::output_iterator< decltype(out_it), std::ranges::range_value_t< decltype(data) > > and
                 std::contiguous_iterator< decltype(out_it) >
    {
        const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
        detail::handleMPIError(MPI_Allreduce(buf_begin, std::addressof(*out_it), buf_size, datatype, op, m_comm),
                               "MPI all-reduce failed");
    }
    void gather(detail::MpiBuf_c auto&& data, auto out_it, int root) const
        requires std::output_iterator< decltype(out_it), std::ranges::range_value_t< decltype(data) > > and
                 std::contiguous_iterator< decltype(out_it) >
    {
        const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
        detail::handleMPIError(
            MPI_Gather(buf_begin, buf_size, datatype, std::addressof(*out_it), buf_size, datatype, root, m_comm),
            "MPI gather failed");
    }
    void broadcast(detail::MpiBuf_c auto&& data, int root) const
    {
        const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
        detail::handleMPIError(MPI_Bcast(buf_begin, buf_size, datatype, root, m_comm), "MPI broadcast failed");
    }
    [[nodiscard]] auto broadcastAsync(detail::MpiBorrowedBuf_c auto&& data, int root) const -> Request
    {
        const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
        Request request{};
        detail::handleMPIError(MPI_Ibcast(buf_begin, buf_size, datatype, root, m_comm, &request.m_request),
                               "MPI asynchronous broadcast failed");
        return request;
    }

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
    MpiComm() = default;
    inline void freeOwnedComm();

    MPI_Comm m_comm = MPI_COMM_NULL;
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
auto MpiComm::FileHandle::readAtAsync(detail::MpiBorrowedBuf_c auto&& read_range, MPI_Offset offset) const
    -> MpiComm::Request
{
    const auto       datatype = detail::MpiType< std::ranges::range_value_t< decltype(read_range) > >::value();
    MpiComm::Request request;
    detail::handleMPIError(
        MPI_File_iread_at(
            m_file, offset, std::ranges::data(read_range), std::ranges::size(read_range), datatype, &request.m_request),
        "MPI asynchronous read from file at offset failed");
    return request;
}
auto MpiComm::FileHandle::writeAtAsync(detail::MpiBorrowedBuf_c auto&& write_range, MPI_Offset offset) const
    -> MpiComm::Request
{
    const auto       datatype = detail::MpiType< std::ranges::range_value_t< decltype(write_range) > >::value();
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

void MpiComm::send(detail::MpiBuf_c auto&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(send_range);
    detail::handleMPIError(MPI_Send(buf_begin, buf_size, datatype, dest, tag, m_comm), "MPI send failed");
}

auto MpiComm::sendAsync(detail::MpiBorrowedBuf_c auto&& data, int dest, int tag) const -> Request
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
    Request request;
    detail::handleMPIError(MPI_Isend(buf_begin, buf_size, datatype, dest, tag, m_comm, &request.m_request),
                           "MPI asynchronous send failed");
    return request;
}

void MpiComm::receive(detail::MpiBuf_c auto&& recv_range, int source, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(recv_range);
    detail::handleMPIError(MPI_Recv(buf_begin, buf_size, datatype, source, tag, m_comm, MPI_STATUS_IGNORE),
                           "MPI receive failed");
}

auto MpiComm::receiveAsync(detail::MpiBorrowedBuf_c auto&& data, int src, int tag) const -> Request

{
    const auto [datatype, buf_begin, buf_size] = detail::decomposeMpiBuf(data);
    Request request;
    detail::handleMPIError(MPI_Irecv(buf_begin, buf_size, datatype, src, tag, m_comm, &request.m_request),
                           "MPI asynchronous receive failed");
    return request;
}

MpiComm MpiComm::distGraphCreate(ContiguousSizedRangeOf< int > auto&& sources,
                                 ContiguousSizedRangeOf< int > auto&& degrees,
                                 ContiguousSizedRangeOf< int > auto&& destinations,
                                 ContiguousSizedRangeOf< int > auto&& weights,
                                 bool                                 reorder) const
{
    MpiComm retval;
    detail::handleMPIError(MPI_Dist_graph_create(m_comm,
                                                 static_cast< int >(std::ranges::ssize(sources)),
                                                 std::ranges::data(sources),
                                                 std::ranges::data(degrees),
                                                 std::ranges::data(destinations),
                                                 std::ranges::data(weights),
                                                 MPI_INFO_NULL,
                                                 reorder,
                                                 &retval.m_comm),
                           "MPI_Dist_graph_create failed");
    return retval;
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

void MpiComm::freeOwnedComm()
{
    if (m_comm != MPI_COMM_NULL)
        MPI_Comm_free(&m_comm);
}

MpiComm::MpiComm(MPI_Comm comm)
{
    detail::handleMPIError(MPI_Comm_dup(comm, &m_comm), "MPI_Comm_dup failed");
}

MpiComm& MpiComm::operator=(MpiComm&& other) noexcept
{
    if (this != &other)
    {
        freeOwnedComm();
        m_comm = std::exchange(other.m_comm, MPI_COMM_NULL);
    }
    return *this;
}
} // namespace lstr
#endif // L3STER_COMM_MPICOMM_HPP
