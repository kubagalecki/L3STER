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
L3STER_MPI_TYPE_MAPPING_STRUCT(std::byte, MPI_BYTE)                        // NOLINT

inline void
handleMPIError(int error, std::string_view err_msg, std::source_location src_loc = std::source_location::current())
{
    util::throwingAssert(not error, err_msg, src_loc);
}

#define L3STER_INVOKE_MPI(fun__, ...) comm::handleMPIError(fun__(__VA_ARGS__), "Call to " #fun__ " failed")

template < typename T >
concept MpiBuiltinType_c = requires { MpiType< std::remove_cvref_t< T > >::value(); };

template < typename T >
concept MpiBuf_c = std::ranges::contiguous_range< T > and std::ranges::sized_range< T > and
                   (MpiBuiltinType_c< std::ranges::range_value_t< T > > or
                    std::is_trivially_copyable_v< std::ranges::range_value_t< T > >);
template < typename T >
concept MpiBorrowedBuf_c = MpiBuf_c< T > and std::ranges::borrowed_range< T >;

template < typename It, typename Buf >
concept MpiOutputIterator_c =
    std::output_iterator< It, std::ranges::range_value_t< Buf > > and
    std::same_as< std::iter_value_t< It >, std::ranges::range_value_t< Buf > > and std::contiguous_iterator< It >;

template < MpiBuiltinType_c T >
struct MpiBufView
{
    MPI_Datatype type;
    T*           data;
    int          size;
};

template < MpiBuf_c Buffer >
auto parseMpiBuf(Buffer&& buf)
{
    using range_value_t = std::remove_reference_t< std::ranges::range_value_t< Buffer > >;
    if constexpr (MpiBuiltinType_c< range_value_t >)
    {
        const auto mpi_type = MpiType< range_value_t >::value();
        const auto data_ptr = std::ranges::data(buf);
        const auto size     = static_cast< int >(std::ranges::size(buf));
        return MpiBufView{mpi_type, data_ptr, size};
    }
    else
    {
        constexpr bool is_const_range =
            std::is_const_v< std::remove_reference_t< decltype(*std::ranges::begin(buf)) > >;
        if constexpr (is_const_range)
            return parseMpiBuf(std::as_bytes(std::span{buf}));
        else
            return parseMpiBuf(std::as_writable_bytes(std::span{buf}));
    }
}
} // namespace comm

class MpiComm
{
public:
    class Status
    {
    public:
        friend class MpiComm;

        template < comm::MpiBuiltinType_c T >
        [[nodiscard]] auto numElems() const -> int;
        template < typename T >
        [[nodiscard]] auto numElems() const -> int
            requires(not comm::MpiBuiltinType_c< T > and std::is_trivially_copyable_v< T >);

        [[nodiscard]] int getSource() const { return m_status.MPI_SOURCE; }
        [[nodiscard]] int getTag() const { return m_status.MPI_TAG; }
        [[nodiscard]] int getError() const { return m_status.MPI_ERROR; }

    private:
        MPI_Status m_status;
    };
    static_assert(std::is_standard_layout_v< Status >);

    class Request
    {
    public:
        friend class MpiComm;

        Request()                          = default;
        Request(const Request&)            = delete;
        Request& operator=(const Request&) = delete;
        inline Request(Request&&) noexcept;
        inline Request& operator=(Request&&) noexcept;
        ~Request() { waitImpl(); }

        void                      wait() { comm::handleMPIError(waitImpl(), "MPI_wait failed"); }
        void                      cancel() { L3STER_INVOKE_MPI(MPI_Cancel, &m_request); }
        [[nodiscard]] inline bool test();

        template < std::ranges::range RequestRange >
        static void waitAll(RequestRange&& requests)
            requires(not std::ranges::contiguous_range< RequestRange > and
                     std::same_as< std::ranges::range_reference_t< RequestRange >, Request& >);
        template < std::ranges::contiguous_range RequestRange >
        static void waitAll(RequestRange&& requests)
            requires std::same_as< std::ranges::range_value_t< RequestRange >, Request >;
        template < std::ranges::range RequestRange >
        static auto waitAny(RequestRange&& requests) -> std::ranges::iterator_t< RequestRange >
            requires(not std::ranges::contiguous_range< RequestRange > and
                     std::same_as< std::ranges::range_reference_t< RequestRange >, Request& >);
        template < std::ranges::contiguous_range RequestRange >
        static auto waitAny(RequestRange&& requests) -> std::ranges::iterator_t< RequestRange >
            requires std::same_as< std::ranges::range_value_t< RequestRange >, Request >;

    private:
        int waitImpl() noexcept { return MPI_Wait(&m_request, MPI_STATUS_IGNORE); }

        MPI_Request m_request = MPI_REQUEST_NULL;
    };
    static_assert(std::is_standard_layout_v< Request >);

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
    [[nodiscard]] inline auto probe(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const -> Status;
    [[nodiscard]] inline auto probeAsync(int source = MPI_ANY_SOURCE, int tag = MPI_ANY_TAG) const
        -> std::pair< Status, bool >;

    // collectives
    void barrier() const { L3STER_INVOKE_MPI(MPI_Barrier, m_comm); }
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void reduce(Data&& data, It out_it, int root, MPI_Op op) const;
    template < comm::MpiBuf_c Data >
    void reduceInPlace(Data&& data, int root, MPI_Op op) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void allReduce(Data&& data, It out_it, MPI_Op op) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void gather(Data&& data, It out_it, int root) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void allGather(Data&& data, It out_it) const;
    template < comm::MpiBuf_c Data >
    void broadcast(Data&& data, int root) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void inclusiveScan(Data&& data, It out_it, MPI_Op op) const;
    template < comm::MpiBuf_c Data >
    void inclusiveScanInPlace(Data&& data, MPI_Op op) const;
    template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
    void exclusiveScan(Data&& data, It out_it, MPI_Op op) const;
    template < comm::MpiBuf_c Data >
    void exclusiveScanInPlace(Data&& data, MPI_Op op) const;

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

private:
    MpiComm() = default;
    inline void freeOwnedComm();

    MPI_Comm m_comm = MPI_COMM_NULL;
};

template < comm::MpiBuiltinType_c T >
auto MpiComm::Status::numElems() const -> int
{
    int retval{};
    L3STER_INVOKE_MPI(MPI_Get_elements, &m_status, comm::MpiType< T >::value(), &retval);
    return retval;
}

template < typename T >
auto MpiComm::Status::numElems() const -> int
    requires(not comm::MpiBuiltinType_c< T > and std::is_trivially_copyable_v< T >)
{
    int retval{};
    L3STER_INVOKE_MPI(MPI_Get_elements, &m_status, MPI_BYTE, &retval);
    retval /= static_cast< int >(sizeof(T));
    return retval;
}

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
    L3STER_INVOKE_MPI(MPI_Test, &m_request, &flag, MPI_STATUS_IGNORE);
    return flag;
}

template < std::ranges::range RequestRange >
void MpiComm::Request::waitAll(RequestRange&& requests)
    requires(not std::ranges::contiguous_range< RequestRange > and
             std::same_as< std::ranges::range_reference_t< RequestRange >, Request& >)
{
    const auto n_requests   = std::ranges::distance(requests);
    auto       raw_requests = util::ArrayOwner< MPI_Request >(static_cast< size_t >(n_requests));
    std::ranges::transform(requests, raw_requests.begin(), [](const Request& r) { return r.m_request; });
    L3STER_INVOKE_MPI(MPI_Waitall, static_cast< int >(n_requests), raw_requests.data(), MPI_STATUSES_IGNORE);
    for (Request& r : requests)
        r.m_request = MPI_REQUEST_NULL;
}

template < std::ranges::contiguous_range RequestRange >
void MpiComm::Request::waitAll(RequestRange&& requests)
    requires std::same_as< std::ranges::range_value_t< RequestRange >, Request >
{
    const auto n_requests       = std::ranges::distance(requests);
    const auto raw_requests_ptr = reinterpret_cast< MPI_Request* >(std::ranges::data(requests)); // standard layout
    L3STER_INVOKE_MPI(MPI_Waitall, static_cast< int >(n_requests), raw_requests_ptr, MPI_STATUSES_IGNORE);
}

template < std::ranges::range RequestRange >
auto MpiComm::Request::waitAny(RequestRange&& requests) -> std::ranges::iterator_t< RequestRange >
    requires(not std::ranges::contiguous_range< RequestRange > and
             std::same_as< std::ranges::range_reference_t< RequestRange >, Request& >)
{
    const auto n_requests   = std::ranges::distance(requests);
    auto       raw_requests = util::ArrayOwner< MPI_Request >(static_cast< size_t >(n_requests));
    std::ranges::transform(requests, raw_requests.begin(), [](const Request& r) { return r.m_request; });
    int index{};
    L3STER_INVOKE_MPI(MPI_Waitany, static_cast< int >(n_requests), raw_requests.data(), &index, MPI_STATUSES_IGNORE);
    if (index == MPI_UNDEFINED)
        return std::ranges::end(requests);
    else
    {
        auto retval       = std::next(std::ranges::begin(requests), index);
        retval->m_request = MPI_REQUEST_NULL;
        return retval;
    }
}

template < std::ranges::contiguous_range RequestRange >
auto MpiComm::Request::waitAny(RequestRange&& requests) -> std::ranges::iterator_t< RequestRange >
    requires std::same_as< std::ranges::range_value_t< RequestRange >, Request >
{
    const auto n_requests       = std::ranges::distance(requests);
    const auto raw_requests_ptr = reinterpret_cast< MPI_Request* >(std::ranges::data(requests)); // standard layout
    int        index{};
    L3STER_INVOKE_MPI(MPI_Waitany, static_cast< int >(n_requests), raw_requests_ptr, &index, MPI_STATUSES_IGNORE);
    return index == MPI_UNDEFINED ? std::ranges::end(requests) : std::next(std::ranges::begin(requests), index);
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
    L3STER_INVOKE_MPI(MPI_File_preallocate, m_file, size);
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::FileHandle::readAtAsync(Data&& read_range, MPI_Offset offset) const -> MpiComm::Request
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(read_range);
    MpiComm::Request request;
    L3STER_INVOKE_MPI(MPI_File_iread_at, m_file, offset, buf_begin, buf_size, datatype, &request.m_request);
    return request;
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::FileHandle::writeAtAsync(Data&& write_range, MPI_Offset offset) const -> MpiComm::Request
{
    const auto datatype = comm::MpiType< std::ranges::range_value_t< decltype(write_range) > >::value();
    auto       request  = MpiComm::Request{};
    L3STER_INVOKE_MPI(MPI_File_iwrite_at,
                      m_file,
                      offset,
                      std::ranges::data(write_range),
                      util::exactIntegerCast< int >(std::ranges::size(write_range)),
                      datatype,
                      &request.m_request);
    return request;
}

MpiComm::FileHandle MpiComm::openFile(const char* file_name, int amode, MPI_Info info) const
{
    auto fh = FileHandle{};
    L3STER_INVOKE_MPI(MPI_File_open, m_comm, file_name, amode, info, &fh.m_file);
    return fh;
}

template < comm::MpiBuf_c Data >
void MpiComm::send(Data&& send_range, int dest, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(send_range);
    L3STER_INVOKE_MPI(MPI_Send, buf_begin, buf_size, datatype, dest, tag, m_comm);
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::sendAsync(Data&& data, int dest, int tag) const -> Request
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    auto request                               = Request{};
    L3STER_INVOKE_MPI(MPI_Isend, buf_begin, buf_size, datatype, dest, tag, m_comm, &request.m_request);
    return request;
}

template < comm::MpiBuf_c Data >
void MpiComm::receive(Data&& recv_range, int src, int tag) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(recv_range);
    L3STER_INVOKE_MPI(MPI_Recv, buf_begin, buf_size, datatype, src, tag, m_comm, MPI_STATUS_IGNORE);
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::receiveAsync(Data&& data, int src, int tag) const -> Request

{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    auto request                               = Request{};
    L3STER_INVOKE_MPI(MPI_Irecv, buf_begin, buf_size, datatype, src, tag, m_comm, &request.m_request);
    return request;
}

auto MpiComm::probe(int source, int tag) const -> Status
{
    auto retval = Status{};
    L3STER_INVOKE_MPI(MPI_Probe, source, tag, m_comm, reinterpret_cast< MPI_Status* >(&retval));
    return retval;
}

auto MpiComm::probeAsync(int source, int tag) const -> std::pair< Status, bool >
{
    auto retval = std::pair< Status, bool >{};
    int  flag{};
    L3STER_INVOKE_MPI(MPI_Iprobe, source, tag, m_comm, &flag, &retval.first.m_status);
    retval.second = flag;
    return retval;
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::reduce(Data&& data, It out_it, int root, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Reduce, buf_begin, std::addressof(*out_it), buf_size, datatype, op, root, m_comm);
}

template < comm::MpiBuf_c Data >
void MpiComm::reduceInPlace(Data&& data, int root, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    getRank() == root ? L3STER_INVOKE_MPI(MPI_Reduce, MPI_IN_PLACE, buf_begin, buf_size, datatype, op, root, m_comm)
                      : L3STER_INVOKE_MPI(MPI_Reduce, buf_begin, nullptr, buf_size, datatype, op, root, m_comm);
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::allReduce(Data&& data, It out_it, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Allreduce, buf_begin, std::addressof(*out_it), buf_size, datatype, op, m_comm);
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::gather(Data&& data, It out_it, int root) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    const auto out_ptr                         = std::addressof(*out_it);
    L3STER_INVOKE_MPI(MPI_Gather, buf_begin, buf_size, datatype, out_ptr, buf_size, datatype, root, m_comm);
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::allGather(Data&& data, It out_it) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    const auto out_ptr                         = std::addressof(*out_it);
    L3STER_INVOKE_MPI(MPI_Allgather, buf_begin, buf_size, datatype, out_ptr, buf_size, datatype, m_comm);
}

template < comm::MpiBuf_c Data >
void MpiComm::broadcast(Data&& data, int root) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Bcast, buf_begin, buf_size, datatype, root, m_comm);
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::inclusiveScan(Data&& data, It out_it, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Scan, buf_begin, std::addressof(*out_it), buf_size, datatype, op, m_comm);
}

template < comm::MpiBuf_c Data >
void MpiComm::inclusiveScanInPlace(Data&& data, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Scan, MPI_IN_PLACE, buf_begin, buf_size, datatype, op, m_comm);
}

template < comm::MpiBuf_c Data, comm::MpiOutputIterator_c< Data > It >
void MpiComm::exclusiveScan(Data&& data, It out_it, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Exscan, buf_begin, std::addressof(*out_it), buf_size, datatype, op, m_comm);
}

template < comm::MpiBuf_c Data >
void MpiComm::exclusiveScanInPlace(Data&& data, MPI_Op op) const
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    L3STER_INVOKE_MPI(MPI_Exscan, MPI_IN_PLACE, buf_begin, buf_size, datatype, op, m_comm);
}

template < comm::MpiBorrowedBuf_c Data >
auto MpiComm::broadcastAsync(Data&& data, int root) const -> MpiComm::Request
{
    const auto [datatype, buf_begin, buf_size] = comm::parseMpiBuf(data);
    auto request                               = Request{};
    L3STER_INVOKE_MPI(MPI_Ibcast, buf_begin, buf_size, datatype, root, m_comm, &request.m_request);
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
    L3STER_INVOKE_MPI(
        MPI_Ialltoall, send_begin, n_elems, send_type, recv_begin, n_elems, recv_type, m_comm, &request.m_request);
    return request;
}

template < ContiguousSizedRangeOf< int > Src,
           ContiguousSizedRangeOf< int > Deg,
           ContiguousSizedRangeOf< int > Dest,
           ContiguousSizedRangeOf< int > Wgts >
MpiComm MpiComm::distGraphCreate(Src&& sources, Deg&& degrees, Dest&& destinations, Wgts&& weights, bool reorder) const
{
    auto retval = MpiComm{};
    L3STER_INVOKE_MPI(MPI_Dist_graph_create,
                      m_comm,
                      static_cast< int >(std::ranges::ssize(sources)),
                      std::ranges::data(sources),
                      std::ranges::data(degrees),
                      std::ranges::data(destinations),
                      std::ranges::data(weights),
                      MPI_INFO_NULL,
                      reorder,
                      &retval.m_comm);
    return retval;
}

int MpiComm::getRank() const
{
    int rank{};
    L3STER_INVOKE_MPI(MPI_Comm_rank, m_comm, &rank);
    return rank;
}

int MpiComm::getSize() const
{
    int size{};
    L3STER_INVOKE_MPI(MPI_Comm_size, m_comm, &size);
    return size;
}

void MpiComm::freeOwnedComm()
{
    if (m_comm != MPI_COMM_NULL)
        MPI_Comm_free(&m_comm);
}

MpiComm::MpiComm(MPI_Comm comm, MPI_Errhandler err_handler)
{
    L3STER_INVOKE_MPI(MPI_Comm_dup, comm, &m_comm);
    L3STER_INVOKE_MPI(MPI_Comm_set_errhandler, m_comm, err_handler);
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
