#ifndef L3STER_COMM_MPICOMM_HPP
#define L3STER_COMM_MPICOMM_HPP

#include "l3ster/util/ArrayOwner.hpp"
#include "l3ster/util/Common.hpp"

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
namespace comm
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

inline void
handleMPIError(int error, std::string_view err_msg, std::source_location src_loc = std::source_location::current())
{
    util::throwingAssert(not error, err_msg, src_loc);
}

template < typename T >
concept MpiType_c = requires { MpiType< std::remove_cvref_t< T > >::value(); };

template < typename T >
concept MpiBuf_c = std::ranges::contiguous_range< T > and std::ranges::sized_range< T > and
                   MpiType_c< std::ranges::range_value_t< T > >;
template < typename T >
concept MpiBorrowedBuf_c = MpiBuf_c< T > and std::ranges::borrowed_range< T >;

template < typename It, typename Buf >
concept MpiOutputIterator_c =
    std::output_iterator< It, std::ranges::range_value_t< Buf > > and std::contiguous_iterator< It >;

template < MpiType_c T >
struct MpiBufView
{
    MPI_Datatype type;
    T*           data;
    int          size;
};

template < MpiBuf_c Buffer >
auto parseMpiBuf(Buffer&& buf)
{
    return MpiBufView{MpiType< std::ranges::range_value_t< decltype(buf) > >::value(),
                      std::ranges::data(buf),
                      static_cast< int >(std::ranges::ssize(buf))};
}
} // namespace comm

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
        ~Request() { waitImpl(); }

        void                      wait() { comm::handleMPIError(waitImpl(), "MPI_wait failed"); }
        void                      cancel() { comm::handleMPIError(MPI_Cancel(&m_request), "MPI_Cancel failed"); }
        [[nodiscard]] inline bool test();

        template < std::ranges::range RequestRange >
        static void waitAll(RequestRange&& requests)
            requires std::same_as< std::ranges::range_reference_t< RequestRange >, Request& >;

    private:
        Request() = default;
        int waitImpl() noexcept { return MPI_Wait(&m_request, MPI_STATUS_IGNORE); }

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
        template < comm::MpiBorrowedBuf_c Data >
        auto readAtAsync(Data&& read_range, MPI_Offset offset) const -> MpiComm::Request;
        template < comm::MpiBorrowedBuf_c Data >
        auto writeAtAsync(Data&& write_range, MPI_Offset offset) const -> MpiComm::Request;

    private:
        FileHandle() = default;
        void closeIgnoreErr() { MPI_File_close(&m_file); }

        MPI_File m_file = MPI_FILE_NULL;
    };

    inline MpiComm(MPI_Comm comm, MPI_Errhandler err_handler = MPI_ERRORS_ARE_FATAL);
    MpiComm(const MpiComm&)            = delete;
    MpiComm& operator=(const MpiComm&) = delete;
    MpiComm(MpiComm&& other) noexcept : m_comm{std::exchange(other.m_comm, MPI_COMM_NULL)} {}
    inline MpiComm& operator=(MpiComm&& other) noexcept;
    ~MpiComm() { freeOwnedComm(); }

    // send
    template < comm::MpiBuf_c Data >
    void send(Data&& send_range, int dest, int tag) const;
    template < comm::MpiBorrowedBuf_c Data >
    [[nodiscard]] auto sendAsync(Data&& data, int dest, int tag) const -> Request;

    // receive
    template < comm::MpiBuf_c Data >
    void receive(Data&& recv_range, int src = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const;
    template < comm::MpiBorrowedBuf_c Data >
    [[nodiscard]] auto receiveAsync(Data&& data, int src = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const -> Request;
    [[nodiscard]] inline auto probe(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const -> MPI_Status;

    // collectives
    void barrier() const { comm::handleMPIError(MPI_Barrier(m_comm), "MPI_Barrier failed"); }
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void reduce(Data&& data, It out_it, int root, MPI_Op op) const;
    template < comm::MpiBuf_c Data >
    void reduceInPlace(Data&& data, int root, MPI_Op op) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void allReduce(Data&& data, It out_it, MPI_Op op) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void gather(Data&& data, It out_it, int root) const;
    template < comm::MpiBuf_c Data >
    void broadcast(Data&& data, int root) const;

    template < comm::MpiBorrowedBuf_c Data >
    [[nodiscard]] auto broadcastAsync(Data&& data, int root) const -> Request;
    template < comm::MpiBorrowedBuf_c SendBuf, comm::MpiBorrowedBuf_c RecvBuf >
    [[nodiscard]] auto allToAllAsync(SendBuf&& send_buf, RecvBuf&& recv_buf) const -> Request;

    // observers
    [[nodiscard]] inline int getRank() const;
    [[nodiscard]] inline int getSize() const;
    [[nodiscard]] MPI_Comm   get() const { return m_comm; }

    // topology-related
    template < ContiguousSizedRangeOf< int > Src,
               ContiguousSizedRangeOf< int > Deg,
               ContiguousSizedRangeOf< int > Dest,
               ContiguousSizedRangeOf< int > Wgts >
    [[nodiscard]] MpiComm
    distGraphCreate(Src&& sources, Deg&& degrees, Dest&& destinations, Wgts&& weights, bool reorder) const;

    // misc
    inline FileHandle openFile(const char* file_name, int amode, MPI_Info info = MPI_INFO_NULL) const;
    void              abort() const { MPI_Abort(m_comm, MPI_ERR_UNKNOWN); }
    template < comm::MpiType_c T >
    static int countReceivedElems(const MPI_Status& status);

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
    waitImpl();
    m_request = std::exchange(other.m_request, MPI_REQUEST_NULL);
    return *this;
}

bool MpiComm::Request::test()
{
    int flag{};
    comm::handleMPIError(MPI_Test(&m_request, &flag, MPI_STATUS_IGNORE), "MPI_Test failed");
    return flag;
}

template < std::ranges::range RequestRange >
void MpiComm::Request::waitAll(RequestRange&& requests)
    requires std::same_as< std::ranges::range_reference_t< RequestRange >, Request& >
{
    const auto n_requests   = std::ranges::distance(requests);
    auto       raw_requests = util::ArrayOwner< MPI_Request >(static_cast< size_t >(n_requests));
    std::ranges::transform(requests, raw_requests.begin(), [](const Request& r) { return r.m_request; });
    comm::handleMPIError(MPI_Waitall(static_cast< int >(n_requests), raw_requests.data(), MPI_STATUSES_IGNORE),
                         "MPI_Waitall failed");
    for (Request& r : requests)
        r.m_request = MPI_REQUEST_NULL;
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
    comm::handleMPIError(MPI_File_preallocate(m_file, size), "MPI_File_preallocate failed");
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::FileHandle::readAtAsync(Data&& read_range, MPI_Offset offset) const -> MpiComm::Request
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(read_range);
    MpiComm::Request request;
    comm::handleMPIError(MPI_File_iread_at(m_file, offset, buf_begin, buf_size, datatype, &request.m_request),
                         "MPI_File_iread_at failed");
    return request;
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::FileHandle::writeAtAsync(Data&& write_range, MPI_Offset offset) const -> MpiComm::Request
{
    const auto datatype = comm::MpiType< std::ranges::range_value_t< decltype(write_range) > >::value();
    auto       request  = MpiComm::Request{};
    comm::handleMPIError(MPI_File_iwrite_at(m_file,
                                            offset,
                                            std::ranges::data(write_range),
                                            util::exactIntegerCast< int >(std::ranges::size(write_range)),
                                            datatype,
                                            &request.m_request),
                         "MPI_File_iwrite_at failed");
    return request;
}

MpiComm::FileHandle MpiComm::openFile(const char* file_name, int amode, MPI_Info info) const
{
    auto fh = FileHandle{};
    comm::handleMPIError(MPI_File_open(m_comm, file_name, amode, info, &fh.m_file), "MPI_File_open failed");
    return fh;
}

template < comm::MpiBuf_c Data >
void MpiComm::send(Data&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(send_range);
    comm::handleMPIError(MPI_Send(buf_begin, buf_size, datatype, dest, tag, m_comm), "MPI_Send failed");
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::sendAsync(Data&& data, int dest, int tag) const -> Request
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    auto request                               = Request{};
    comm::handleMPIError(MPI_Isend(buf_begin, buf_size, datatype, dest, tag, m_comm, &request.m_request),
                         "MPI_Isend failed");
    return request;
}

template < comm::MpiBuf_c Data >
void MpiComm::receive(Data&& recv_range, int src, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(recv_range);
    comm::handleMPIError(MPI_Recv(buf_begin, buf_size, datatype, src, tag, m_comm, MPI_STATUS_IGNORE),
                         "MPI_Recv failed");
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::receiveAsync(Data&& data, int src, int tag) const -> Request

{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    auto request                               = Request{};
    comm::handleMPIError(MPI_Irecv(buf_begin, buf_size, datatype, src, tag, m_comm, &request.m_request),
                         "MPI_Irecv failed");
    return request;
}

auto MpiComm::probe(int source, int tag) const -> MPI_Status
{
    auto retval = MPI_Status{};
    comm::handleMPIError(MPI_Probe(source, tag, m_comm, &retval), "MPI_Probe failed");
    return retval;
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::reduce(Data&& data, It out_it, int root, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    comm::handleMPIError(MPI_Reduce(buf_begin, std::addressof(*out_it), buf_size, datatype, op, root, m_comm),
                         "MPI_Reduce failed");
}

template < comm::MpiBuf_c Data >
void MpiComm::reduceInPlace(Data&& data, int root, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    comm::handleMPIError(getRank() == root ? MPI_Reduce(MPI_IN_PLACE, buf_begin, buf_size, datatype, op, root, m_comm)
                                           : MPI_Reduce(buf_begin, nullptr, buf_size, datatype, op, root, m_comm),
                         "MPI_Reduce failed");
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::allReduce(Data&& data, It out_it, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    comm::handleMPIError(MPI_Allreduce(buf_begin, std::addressof(*out_it), buf_size, datatype, op, m_comm),
                         "MPI_Allreduce failed");
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::gather(Data&& data, It out_it, int root) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    comm::handleMPIError(
        MPI_Gather(buf_begin, buf_size, datatype, std::addressof(*out_it), buf_size, datatype, root, m_comm),
        "MPI_Gather failed");
}

template < comm::MpiBuf_c Data >
void MpiComm::broadcast(Data&& data, int root) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    comm::handleMPIError(MPI_Bcast(buf_begin, buf_size, datatype, root, m_comm), "MPI_Bcast failed");
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::broadcastAsync(Data&& data, int root) const -> MpiComm::Request
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    auto request                               = Request{};
    comm::handleMPIError(MPI_Ibcast(buf_begin, buf_size, datatype, root, m_comm, &request.m_request),
                         "MPI_Ibcast failed");
    return request;
}

template < comm::MpiBorrowedBuf_c SendBuf, comm::MpiBorrowedBuf_c RecvBuf >
auto MpiComm::allToAllAsync(SendBuf&& send_buf, RecvBuf&& recv_buf) const -> MpiComm::Request
{
    static_assert(std::same_as< std::remove_cvref< std::ranges::range_value_t< SendBuf > >,
                                std::remove_cvref< std::ranges::range_value_t< RecvBuf > > >,
                  "The send and receive buffers of the all-to-all collective must have the same type");
    const auto [send_type, send_begin, send_size] = comm::parseMpiBuf(send_buf);
    const auto [recv_type, recv_begin, recv_size] = comm::parseMpiBuf(recv_buf);
    util::throwingAssert(send_size == recv_size,
                         "The send and receive buffers of the all-to-all collective must have the same size");

    const int n_elems = send_size / getSize();
    auto      request = Request{};
    comm::handleMPIError(
        MPI_Ialltoall(send_begin, n_elems, send_type, recv_begin, n_elems, recv_type, m_comm, &request.m_request),
        "MPI_IAlltoall failed");
    return request;
}

template < ContiguousSizedRangeOf< int > Src,
           ContiguousSizedRangeOf< int > Deg,
           ContiguousSizedRangeOf< int > Dest,
           ContiguousSizedRangeOf< int > Wgts >
MpiComm MpiComm::distGraphCreate(Src&& sources, Deg&& degrees, Dest&& destinations, Wgts&& weights, bool reorder) const
{
    auto retval = MpiComm{};
    comm::handleMPIError(MPI_Dist_graph_create(m_comm,
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

template < comm::MpiType_c T >
int MpiComm::countReceivedElems(const MPI_Status& status)
{
    int retval{};
    comm::handleMPIError(MPI_Get_count(&status, comm::MpiType< T >::value(), &retval), "MPI_Get_count failed");
    return retval;
}

int MpiComm::getRank() const
{
    int rank{};
    comm::handleMPIError(MPI_Comm_rank(m_comm, &rank), "MPI_Comm_rank failed");
    return rank;
}

int MpiComm::getSize() const
{
    int size{};
    comm::handleMPIError(MPI_Comm_size(m_comm, &size), "MPI_Comm_size failed");
    return size;
}

void MpiComm::freeOwnedComm()
{
    if (m_comm != MPI_COMM_NULL)
        MPI_Comm_free(&m_comm);
}

MpiComm::MpiComm(MPI_Comm comm, MPI_Errhandler err_handler)
{
    comm::handleMPIError(MPI_Comm_dup(comm, &m_comm), "MPI_Comm_dup failed");
    comm::handleMPIError(MPI_Comm_set_errhandler(m_comm, err_handler), "MPI_Comm_set_errhandler failed");
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
