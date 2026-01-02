#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"
#include "l3ster/util/IndexMap.hpp"

#include <iterator>

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
template < el_o_t... orders, size_t max_dofs_per_node >
auto makeGhostToWgtMap(const mesh::MeshPartition< orders... >&       mesh,
                       const ProblemDefinition< max_dofs_per_node >& problem_def)
    -> robin_hood::unordered_flat_map< n_id_t, int >
{
    auto retval = robin_hood::unordered_flat_map< n_id_t, int >(mesh.getNodeOwnership().shared().size());
    if constexpr (max_dofs_per_node == 0)
        for (auto n : mesh.getNodeOwnership().shared())
            retval[n] = 1;
    else
    {
        const auto is_ghost_node = [&](n_id_t node) {
            return not mesh.getNodeOwnership().isOwned(node);
        };
        auto dof_map = robin_hood::unordered_flat_map< n_id_t, std::bitset< max_dofs_per_node > >{retval.size()};
        for (const auto& [domains, dof_bmp] : problem_def)
        {
            const auto visit_elem = [&](const auto& element) {
                for (auto node : element.nodes | std::views::filter(is_ghost_node))
                    dof_map[node] |= dof_bmp;
            };
            mesh.visit(visit_elem, domains);
        }
        for (auto&& [n, b] : dof_map)
            retval[n] = static_cast< int >(b.count());
    }
    return retval;
}

template < el_o_t... orders >
auto makeNodeDist(const util::ArrayOwner< mesh::MeshPartition< orders... > >& mesh_parts) -> util::ArrayOwner< n_id_t >
{
    auto retval = util::ArrayOwner< n_id_t >(mesh_parts.size());
    std::transform_inclusive_scan(
        mesh_parts.begin(),
        mesh_parts.end(),
        retval.begin(),
        std::plus{},
        [](const mesh::MeshPartition< orders... >& part) { return part.getNodeOwnership().owned().size(); });
    return retval;
}

struct CommVolumeInfo
{
    std::vector< int > sources, degrees, destinations, weights;
};
template < el_o_t... orders, size_t max_dofs_per_node >
auto makeCommVolumeInfo(const util::ArrayOwner< mesh::MeshPartition< orders... > >& mesh_parts,
                        const ProblemDefinition< max_dofs_per_node >&               problem_def) -> CommVolumeInfo
{
    const auto get_node_owner = [node_dist = makeNodeDist(mesh_parts)](n_id_t node) {
        const auto lb_iter = std::ranges::upper_bound(node_dist, node);
        return std::distance(node_dist.begin(), lb_iter);
    };
    auto retval    = CommVolumeInfo{};
    auto dest_wgts = util::ArrayOwner< int >(mesh_parts.size(), 0);
    for (auto&& [part_ind, part] : mesh_parts | std::views::enumerate)
    {
        const auto ghost_to_wgt_map = makeGhostToWgtMap(part, problem_def);
        std::ranges::fill(dest_wgts, 0);
        for (auto node : part.getNodeOwnership().shared())
            if (const auto dof_it = ghost_to_wgt_map.find(node); dof_it != ghost_to_wgt_map.end())
            {
                const auto owner = get_node_owner(node);
                dest_wgts.at(owner) += dof_it->second;
            }
        if (const auto n_dests = std::ranges::count_if(dest_wgts, [](auto wgt) { return wgt > 0; }); n_dests > 0)
        {
            retval.sources.push_back(static_cast< int >(part_ind));
            retval.degrees.push_back(static_cast< int >(n_dests));
            for (auto&& [dest_ind, dest_wgt] : dest_wgts | std::views::enumerate)
                if (dest_wgt > 0)
                {
                    retval.destinations.push_back(static_cast< int >(dest_ind));
                    retval.weights.push_back(dest_wgt);
                }
        }
    }
    return retval;
}

template < el_o_t... orders, size_t max_dofs_per_node >
auto computeOptimizedRankPermutation(const MpiComm&                                              comm,
                                     const util::ArrayOwner< mesh::MeshPartition< orders... > >& mesh_parts,
                                     const ProblemDefinition< max_dofs_per_node >&               problem_def)
    -> util::ArrayOwner< int >
{
    const auto& [sources, degrees, dests, wgts] = makeCommVolumeInfo(mesh_parts, problem_def);
    MPI_Comm new_comm                           = MPI_COMM_NULL;
    L3STER_INVOKE_MPI(MPI_Dist_graph_create,
                      comm.get(),
                      static_cast< int >(sources.size()),
                      sources.data(),
                      degrees.data(),
                      dests.data(),
                      wgts.empty() ? MPI_WEIGHTS_EMPTY : wgts.data(),
                      MPI_INFO_NULL,
                      true,
                      &new_comm);
    util::throwingAssert(new_comm != MPI_COMM_NULL);
    int new_rank;
    L3STER_INVOKE_MPI(MPI_Comm_rank, new_comm, &new_rank);
    L3STER_INVOKE_MPI(MPI_Comm_free, &new_comm);
    util::throwingAssert(new_rank >= 0 and new_rank < comm.getSize());
    auto retval = util::ArrayOwner< int >(mesh_parts.size());
    comm.gather(std::views::single(new_rank), retval.begin(), 0);
    return retval;
}

template < el_o_t... orders, size_t max_dofs_per_node >
auto permuteMesh(const MpiComm&                                       comm,
                 util::ArrayOwner< mesh::MeshPartition< orders... > > mesh_parts,
                 const ProblemDefinition< max_dofs_per_node >&        problem_def,
                 const MeshDistOpts& opts) -> util::ArrayOwner< mesh::MeshPartition< orders... > >
{
    if (opts.optimize == false)
        return mesh_parts;
    const auto opt_permutation = computeOptimizedRankPermutation(comm, mesh_parts, problem_def);
    if (mesh_parts.empty())
        return {}; // All ranks need to participate in computing the permutation, but only rank 0 actually performs it
    auto retval = util::ArrayOwner< mesh::MeshPartition< orders... > >(mesh_parts.size());
    for (auto&& [old_ind, new_ind] : opt_permutation | std::views::enumerate)
        retval[new_ind] = std::move(mesh_parts[old_ind]);
    const auto node_reorder_map = util::IndexMap< n_id_t, n_id_t >{
        retval | std::views::transform([](const auto& mesh) { return mesh.getNodeOwnership().owned(); }) |
        std::views::join};
    for (auto& part : retval)
        part.reindexNodes(node_reorder_map);
    return retval;
}

template < el_o_t... orders >
auto getOldNodeIds(const mesh::MeshPartition< orders... >&                 mesh_parts,
                   const robin_hood::unordered_flat_map< n_id_t, n_id_t >& new_to_old)
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
auto receiveMesh(const MpiComm& comm, int src_rank, int tag = 0) -> mesh::MeshPartition< orders... >
{
    const auto size       = static_cast< size_t >(comm.probe(src_rank, tag).numElems< char >());
    auto       recv_alloc = util::ArrayOwner< char >(size);
    comm.receive(recv_alloc, src_rank, 0);
    return mesh::deserializeMesh< orders... >(std::string_view{recv_alloc});
}

template < el_o_t... orders >
auto receiveMesh(const MpiComm& comm, int src_rank, util::TypePack< mesh::MeshPartition< orders... > >)
    -> mesh::MeshPartition< orders... >
{
    return receiveMesh< orders... >(comm, src_rank);
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

    const auto node_throughputs = comm::gatherNodeThroughputs(comm);
    const auto mesh             = comm.getRank() == 0 ? std::invoke(mesh_generator) : partition_t{};
    auto       mesh_parted = comm.getRank() == 0 ? partitionMesh(mesh, comm.getSize(), node_throughputs, problem_def)
                                                 : util::ArrayOwner< partition_t >{};
    mesh_parted            = detail::permuteMesh(comm, std::move(mesh_parted), problem_def, opts);

    if (comm.getRank() == 0)
    {
        auto serial_meshes = util::ArrayOwner< std::string >(comm.getSizeUz() - 1);
        util::tbb::parallelTransform(mesh_parted | std::views::drop(1), serial_meshes.begin(), [](const auto& part) {
            return mesh::serializeMesh(part);
        });
        auto reqs = std::views::zip_transform(
                        [&](const std::string& serial, int dest) { return comm.sendAsync(serial, dest, 0); },
                        serial_meshes,
                        std::views::iota(1, comm.getSize())) |
                    std::ranges::to< util::ArrayOwner >();
        MpiComm::Request::waitAll(reqs);
        return std::make_shared< partition_t >(std::move(mesh_parted.front()));
    }
    else
        return std::make_shared< partition_t >(comm::receiveMesh(comm, 0, util::TypePack< partition_t >{}));
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
        auto       node_map  = util::ArrayOwner< n_id_t >{std::views::iota(n_id_t{0}, num_nodes)};
        return std::make_pair(std::move(mesh_ptr), std::move(node_map));
    }

    const auto node_throughputs = gatherNodeThroughputs(comm);
    const auto mesh             = comm.getRank() == 0 ? std::invoke(mesh_generator) : partition_t{};
    auto       node_map         = robin_hood::unordered_flat_map< n_id_t, n_id_t >{};
    auto       mesh_parted      = comm.getRank() == 0
                                    ? partitionMesh(mesh, comm.getSize(), node_map, node_throughputs, problem_def)
                                    : util::ArrayOwner< partition_t >{};
    node_map                    = util::invertMap(node_map);
    mesh_parted                 = detail::permuteMesh(comm, std::move(mesh_parted), problem_def, opts);

    if (comm.getRank() == 0)
    {
        auto serial_meshes = util::ArrayOwner< std::string >(comm.getSizeUz() - 1);
        auto old_ids       = util::ArrayOwner< util::ArrayOwner< n_id_t > >(comm.getSizeUz());
        util::tbb::parallelTransform(mesh_parted | std::views::drop(1), serial_meshes.begin(), [](const auto& part) {
            return mesh::serializeMesh(part);
        });
        util::tbb::parallelTransform(
            mesh_parted, old_ids.begin(), [&](const auto& part) { return detail::getOldNodeIds(part, node_map); });
        auto mesh_reqs = std::views::zip_transform(
                             [&](std::string_view serial, int dest) { return comm.sendAsync(serial, dest, 0); },
                             serial_meshes,
                             std::views::iota(1, comm.getSize())) |
                         std::ranges::to< util::ArrayOwner >();
        auto node_reqs =
            std::views::zip_transform([&](const auto& nodes, int dest) { return comm.sendAsync(nodes, dest, 1); },
                                      old_ids | std::views::drop(1),
                                      std::views::iota(1, comm.getSize())) |
            std::ranges::to< util::ArrayOwner >();
        MpiComm::Request::waitAll(mesh_reqs);
        MpiComm::Request::waitAll(node_reqs);
        return std::make_pair(std::make_shared< partition_t >(std::move(mesh_parted.front())),
                              std::move(old_ids.front()));
    }
    else
    {
        auto       part      = comm::receiveMesh(comm, 0, util::TypePack< partition_t >{});
        const auto num_nodes = part.getNodeOwnership().localSize();
        auto       old_ids   = util::ArrayOwner< n_id_t >(num_nodes);
        comm.receive(old_ids, 0, 1);
        return std::make_pair(std::make_shared< partition_t >(std::move(part)), std::move(old_ids));
    }
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
