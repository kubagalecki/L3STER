#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

namespace lstr
{
struct MeshDistOpts
{
    bool optimize = false;
};

namespace comm
{
namespace detail
{
inline int invertPermutation(const MpiComm& comm, int dest_rank)
{
    constexpr int  tag      = 101;
    constexpr auto ping     = std::array{'\0'};
    const auto     send_req = comm.sendAsync(ping, dest_rank, tag);
    const auto     src_rank = comm.probe(MPI_ANY_SOURCE, tag).getSource();
    auto           _        = ping;
    comm.receive(_, src_rank, tag);
    return src_rank;
}

template < el_o_t... orders >
auto getCommGraph(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh)
    -> std::array< util::CrsGraph< int >, 2 >
{
    const auto            node_dist    = mesh.getNodeOwnership().getOwnershipDist(comm);
    const auto            out_nbr_info = mesh.getNodeOwnership().computeOutNbrInfo(node_dist);
    const auto            num_out_nbrs = out_nbr_info.size();
    const auto            sizes        = std::views::single(num_out_nbrs);
    util::CrsGraph< int > out_graph{sizes}, out_wgt{sizes};
    std::ranges::copy(out_nbr_info | std::views::keys, out_graph(0).begin());
    std::ranges::transform(out_nbr_info | std::views::values, out_wgt(0).begin(), &std::vector< n_id_t >::size);
    return {std::move(out_graph), std::move(out_wgt)};
}

template < el_o_t... orders >
int computeOptimizedRank(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh)
{
    const auto [nbrs, wgts] = getCommGraph(comm, mesh);
    const int my_rank       = comm.getRank();
    return comm.distGraphCreate(std::span{&my_rank, 1}, nbrs, wgts, true).getRank();
}

template < el_o_t... orders >
auto sendRecvMesh(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh, int dest_rank)
    -> std::pair< int, mesh::MeshPartition< orders... > >
{
    constexpr int tag         = 0;
    const auto    serial_send = mesh::serializeMesh(mesh);
    const auto    send_req    = comm.sendAsync(serial_send, dest_rank, tag);
    const auto    recv_status = comm.probe(MPI_ANY_SOURCE, tag);
    const auto    recv_rank   = recv_status.getSource();
    const auto    recv_sz     = recv_status.numElems< char >();
    auto          seral_recv  = util::ArrayOwner< char >(static_cast< size_t >(recv_sz));
    comm.receive(seral_recv, recv_rank, tag);
    return std::make_pair(recv_rank, mesh::deserializeMesh< orders... >(std::string_view{seral_recv}));
}

template < el_o_t... orders >
auto establishNewNodeIds(const MpiComm&                          comm,
                         const mesh::MeshPartition< orders... >& old_mesh,
                         int                                     dest_rank,
                         const mesh::MeshPartition< orders... >& new_mesh,
                         int                                     src_rank) -> util::SegmentedOwnership< n_id_t >
{
    constexpr int tag                              = 1;
    const auto&   new_own                          = new_mesh.getNodeOwnership();
    const auto    new_num_owned                    = new_own.owned().size();
    const auto [new_begin, new_begin_of_old_nodes] = std::invoke([&] {
        n_id_t nb{new_num_owned}, nbon{};
        comm.exclusiveScanInPlace(std::span{&nb, 1}, MPI_SUM);
        nb       = comm.getRank() == 0 ? 0 : nb;
        auto req = comm.receiveAsync(std::span{&nbon, 1}, dest_rank, tag);
        comm.send(std::span{&nb, 1}, src_rank, tag);
        req.wait();
        return std::make_pair(nb, nbon);
    });

    const auto& old_own              = old_mesh.getNodeOwnership();
    const auto  old_num_owned        = old_own.owned().size();
    const auto  context_old          = std::make_shared< ImportExportContext >(comm, old_own);
    auto        importer             = Import< n_id_t >{context_old, 1};
    auto        new_ids_of_old_nodes = util::ArrayOwner< n_id_t >(old_own.localSize());
    std::ranges::iota(new_ids_of_old_nodes | std::views::take(old_num_owned), new_begin_of_old_nodes);
    importer.setOwned(new_ids_of_old_nodes, new_ids_of_old_nodes.size());
    importer.setShared(new_ids_of_old_nodes | std::views::drop(old_num_owned), new_ids_of_old_nodes.size());
    importer.doBlockingImport(comm);

    auto new_shared = util::ArrayOwner< n_id_t >(new_own.shared().size());
    auto req        = comm.receiveAsync(new_shared, src_rank, tag);
    comm.send(new_ids_of_old_nodes | std::views::drop(old_num_owned), dest_rank, tag);
    req.wait();
    return {new_begin, new_num_owned, new_shared};
}

template < el_o_t... orders >
auto permuteMesh(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh, int dest_rank)
    -> mesh::MeshPartition< orders... >
{
    auto [src_rank, new_mesh] = sendRecvMesh(comm, mesh, dest_rank);
    const auto new_own        = establishNewNodeIds(comm, mesh, dest_rank, new_mesh, src_rank);
    const auto old_own        = copy(new_mesh.getNodeOwnership());
    const auto old2new        = [&](n_id_t node) {
        const auto old_local = old_own.getLocalIndex(node);
        return new_own.getGlobalIndex(old_local);
    };
    new_mesh.reindexNodes(old2new);
    return std::move(new_mesh);
}

template < el_o_t... orders >
auto getOldNodeIds(const mesh::MeshPartition< orders... >&                 mesh_parts,
                   const robin_hood::unordered_flat_map< n_id_t, n_id_t >& new_to_old) -> util::ArrayOwner< n_id_t >
{
    const auto& node_own = mesh_parts.getNodeOwnership();
    const auto  to_old   = [&](n_id_t new_id) {
        return new_to_old.at(new_id);
    };
    auto       retval       = util::ArrayOwner< n_id_t >(node_own.localSize());
    const auto shared_begin = std::ranges::transform(node_own.owned(), retval.begin(), to_old).out;
    std::ranges::transform(node_own.shared(), shared_begin, to_old);
    return retval;
}
} // namespace detail

template < el_o_t... orders >
auto receiveMesh(const MpiComm& comm, int src_rank, int tag) -> mesh::MeshPartition< orders... >
{
    const auto size       = static_cast< size_t >(comm.probe(src_rank, tag).numElems< char >());
    auto       recv_alloc = util::ArrayOwner< char >(size);
    comm.receive(recv_alloc, src_rank, tag);
    return mesh::deserializeMesh< orders... >(std::string_view{recv_alloc});
}

template < el_o_t... orders >
auto receiveMesh(const MpiComm& comm, int src_rank, int tag, util::TypePack< mesh::MeshPartition< orders... > >)
    -> mesh::MeshPartition< orders... >
{
    return receiveMesh< orders... >(comm, src_rank, tag);
}

namespace detail
{
template < el_o_t... orders >
auto scatterParts(const MpiComm& comm, std::span< mesh::MeshPartition< orders... > > mesh_parts)
    -> mesh::MeshPartition< orders... >
{
    constexpr auto tag = 0;
    if (comm.getRank() == 0)
    {
        util::throwingAssert(mesh_parts.size() == comm.getSizeUz());
        auto serial = util::ArrayOwner< std::string >(comm.getSizeUz() - 1);
        util::tbb::parallelTransform(mesh_parts | std::views::drop(1), serial.begin(), [](const auto& part) {
            return mesh::serializeMesh(part);
        });
        const auto send_serial = [&](const std::string& serial_part, int dest) {
            return comm.sendAsync(serial_part, dest, tag);
        };
        auto reqs = std::views::zip_transform(send_serial, serial, std::views::iota(1, comm.getSize())) |
                    std::ranges::to< util::ArrayOwner >();
        MpiComm::Request::waitAll(reqs);
        return std::move(mesh_parts.front());
    }
    return receiveMesh(comm, 0, tag, util::TypePack< mesh::MeshPartition< orders... > >{});
}

template < el_o_t... orders >
auto scatterNodes(const MpiComm&                                          comm,
                  std::span< const mesh::MeshPartition< orders... > >     mesh_parts,
                  const robin_hood::unordered_flat_map< n_id_t, n_id_t >& node_map)
{
    constexpr auto tag              = 1;
    const auto     node_throughputs = gatherNodeThroughputs(comm);
    if (comm.getRank() == 0)
    {
        util::throwingAssert(mesh_parts.size() == comm.getSizeUz());
        auto old_ids = util::ArrayOwner< util::ArrayOwner< n_id_t > >(comm.getSizeUz());
        util::tbb::parallelTransform(
            mesh_parts, old_ids.begin(), [&](const auto& part) { return detail::getOldNodeIds(part, node_map); });
        auto reqs =
            std::views::zip_transform([&](const auto& nodes, int dest) { return comm.sendAsync(nodes, dest, tag); },
                                      old_ids | std::views::drop(1),
                                      std::views::iota(1, comm.getSize())) |
            std::ranges::to< util::ArrayOwner >();
        MpiComm::Request::waitAll(reqs);
        return old_ids.front();
    }

    const auto num_nodes = static_cast< size_t >(comm.probe(0, tag).numElems< n_id_t >());
    auto       retval    = util::ArrayOwner< n_id_t >(num_nodes);
    comm.receive(retval, 0, tag);
    return retval;
}
} // namespace detail

template < el_o_t... orders >
auto optimizeMeshDistribution(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh)
    -> mesh::MeshPartition< orders... >
{
    const auto dest_rank = detail::invertPermutation(comm, detail::computeOptimizedRank(comm, mesh));
    return detail::permuteMesh(comm, mesh, dest_rank);
}

template < el_o_t... orders >
auto optimizeMeshDistribution(const MpiComm&                          comm,
                              const mesh::MeshPartition< orders... >& mesh,
                              std::span< const n_id_t >               nodes)
    -> std::pair< mesh::MeshPartition< orders... >, util::ArrayOwner< n_id_t > >
{
    constexpr int tag       = 2;
    const auto    src_rank  = detail::computeOptimizedRank(comm, mesh);
    const auto    dest_rank = detail::invertPermutation(comm, src_rank);
    auto          new_mesh  = detail::permuteMesh(comm, mesh, dest_rank);
    auto          new_nodes = util::shift(comm, nodes, dest_rank, src_rank, tag);
    return std::make_pair(std::move(new_mesh), std::move(new_nodes));
}

template < typename MeshGenerator, size_t max_dofs_per_node = 0 >
auto distributeMesh(const MpiComm&                                comm,
                    MeshGenerator&&                               mesh_generator,
                    const ProblemDefinition< max_dofs_per_node >& problem_def = {},
                    const MeshDistOpts&                           opts        = {})
{
    L3STER_PROFILE_FUNCTION;
    using partition_t = std::remove_cvref_t< decltype(std::invoke(mesh_generator)) >;

    if (comm.getSize() == 1)
        return std::make_shared< partition_t >(std::invoke(mesh_generator));

    const auto node_tp = gatherNodeThroughputs(comm);
    auto parted = comm.getRank() == 0 ? partitionMesh(std::invoke(mesh_generator), comm.getSize(), node_tp, problem_def)
                                      : util::ArrayOwner< partition_t >{};
    auto my_partition = detail::scatterParts(comm, std::span{parted});
    if (opts.optimize)
        my_partition = optimizeMeshDistribution(comm, my_partition);
    return std::make_shared< partition_t >(std::move(my_partition));
}

template < typename MeshGenerator, size_t max_dofs_per_node = 0 >
auto distributeMeshAndNodeMap(const MpiComm&                                comm,
                              MeshGenerator&&                               mesh_generator,
                              const ProblemDefinition< max_dofs_per_node >& problem_def = {},
                              const MeshDistOpts&                           opts        = {})
{
    L3STER_PROFILE_FUNCTION;
    using partition_t = std::remove_cvref_t< decltype(std::invoke(mesh_generator)) >;

    if (comm.getSize() == 1)
    {
        auto       mesh_ptr  = std::make_shared< partition_t >(std::invoke(mesh_generator));
        const auto num_nodes = static_cast< n_id_t >(mesh_ptr->getNodeOwnership().owned().size());
        return std::make_pair(std::move(mesh_ptr),
                              std::views::iota(n_id_t{0}, num_nodes) | std::ranges::to< util::ArrayOwner >());
    }

    const auto node_tp  = gatherNodeThroughputs(comm);
    auto       node_map = robin_hood::unordered_flat_map< n_id_t, n_id_t >{};
    auto       parted   = comm.getRank() == 0
                            ? partitionMesh(std::invoke(mesh_generator), comm.getSize(), node_map, node_tp, problem_def)
                            : util::ArrayOwner< partition_t >{};
    node_map            = util::invertMap(node_map);
    auto old_nodes      = detail::scatterNodes(comm, std::span{std::as_const(parted)}, node_map);
    auto my_partition   = detail::scatterParts(comm, std::span{parted});

    if (opts.optimize)
        std::tie(my_partition, old_nodes) =
            optimizeMeshDistribution(comm, my_partition, std::span{std::as_const(old_nodes)});

    return std::make_pair(std::make_shared< partition_t >(std::move(my_partition)), std::move(old_nodes));
}
} // namespace comm

template < el_o_t order, GeneratorFor_c< mesh::MeshPartition< 1 > > Generator, size_t max_dofs_per_node = 0 >
auto generateAndDistributeMesh(const MpiComm&                                comm,
                               Generator&&                                   mesh_generator,
                               util::ConstexprValue< order >                 order_ctwrpr = {},
                               const ProblemDefinition< max_dofs_per_node >& problem_def  = {},
                               const MeshDistOpts& opts = {}) -> std::shared_ptr< mesh::MeshPartition< order > >
{
    const auto converted_mesh_generator = [&] {
        auto generated_mesh = std::invoke(std::forward< Generator >(mesh_generator));
        if constexpr (order_ctwrpr.value == 1)
            return generated_mesh;
        else
            return convertMeshToOrder< order >(generated_mesh);
    };
    return comm::distributeMesh(comm, converted_mesh_generator, problem_def, opts);
}

template < el_o_t order, mesh::MeshFormat mesh_format, size_t max_dofs_per_node = 0 >
auto readAndDistributeMesh(const MpiComm&                                comm,
                           std::string_view                              mesh_file,
                           mesh::MeshFormatTag< mesh_format >            format_tag,
                           const util::ArrayOwner< d_id_t >&             boundaries,
                           util::ConstexprValue< order >                 order_ctwrpr = {},
                           const ProblemDefinition< max_dofs_per_node >& problem_def  = {},
                           const MeshDistOpts& opts = {}) -> std::shared_ptr< mesh::MeshPartition< order > >
{
    const auto read_generator = [&] {
        return readMesh(mesh_file, boundaries, format_tag);
    };
    return generateAndDistributeMesh(comm, read_generator, order_ctwrpr, problem_def, opts);
}

template < el_o_t order, GeneratorFor_c< mesh::MeshPartition< 1 > > Generator, ProblemDef problem_def >
[[deprecated]] auto generateAndDistributeMesh(const MpiComm&                comm,
                                              Generator&&                   mesh_generator,
                                              util::ConstexprValue< order > order_ctwrpr,
                                              util::ConstexprValue< problem_def >,
                                              const MeshDistOpts& opts = {})
    -> std::shared_ptr< mesh::MeshPartition< order > >
{
    const auto prob_def = detail::convertToRuntime(problem_def);
    return generateAndDistributeMesh(comm, std::forward< Generator >(mesh_generator), order_ctwrpr, prob_def, opts);
}

template < el_o_t order, mesh::MeshFormat mesh_format, ProblemDef problem_def >
[[deprecated]] auto readAndDistributeMesh(const MpiComm&                     comm,
                                          std::string_view                   mesh_file,
                                          mesh::MeshFormatTag< mesh_format > format_tag,
                                          const util::ArrayOwner< d_id_t >&  boundaries,
                                          util::ConstexprValue< order >      order_ctwrpr,
                                          util::ConstexprValue< problem_def >,
                                          const MeshDistOpts& opts = {})
    -> std::shared_ptr< mesh::MeshPartition< order > >
{
    const auto prob_def = detail::convertToRuntime(problem_def);
    return readAndDistributeMesh(comm, mesh_file, format_tag, boundaries, order_ctwrpr, prob_def, opts);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
