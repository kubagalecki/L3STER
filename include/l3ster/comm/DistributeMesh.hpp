#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "DeserializeMesh.hpp"
#include "ReceiveMesh.hpp"
#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

namespace lstr
{
namespace detail::dist_mesh
{
template < detail::ProblemDef_c auto problem_def >
auto makeNodeToDofMap(const MeshPartition& mesh, ConstexprValue< problem_def > probdef_ctwrapper)
    -> robin_hood::unordered_flat_map< n_id_t, std::bitset< deduceNFields(problem_def) > >
{
    auto retval =
        robin_hood::unordered_flat_map< n_id_t, std::bitset< deduceNFields(problem_def) > >{mesh.getAllNodes().size()};
    forConstexpr(
        [&]< auto dom_def >(ConstexprValue< dom_def >) {
            constexpr auto dom_id   = dom_def.first;
            constexpr auto dom_dofs = getTrueInds< dom_def.second >();
            mesh.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                    {
                        auto& node_map_entry = retval[node];
                        for (auto dof : dom_dofs)
                            node_map_entry.set(dof);
                    }
                },
                dom_id);
        },
        probdef_ctwrapper);
    return retval;
}

inline auto makeNodeToPartMap(const Mesh& mesh) -> std::vector< int >
{
    const size_t n_nodes = std::ranges::max(
        mesh.getPartitions() |
        std::views::filter([](const MeshPartition& part) { return not part.getOwnedNodes().empty(); }) |
        std::views::transform([](const MeshPartition& part) { return part.getOwnedNodes().back(); }));
    std::vector< int > retval(n_nodes);
    for (int part_ind = 0; const MeshPartition& part : mesh.getPartitions())
    {
        for (auto node : part.getOwnedNodes())
            retval[node] = part_ind;
        ++part_ind;
    }
    return retval;
}

struct CommVolumeInfo
{
    std::vector< int > sources, degrees, destinations, weights;
};
template < detail::ProblemDef_c auto problem_def >
auto makeCommVolumeInfo(const Mesh& mesh, ConstexprValue< problem_def > probdef_ctwrapper)
{
    const auto node_to_part_map = makeNodeToPartMap(mesh);
    auto       retval           = CommVolumeInfo{};
    for (int part_ind = 0; const MeshPartition& part : mesh.getPartitions())
    {
        const auto node_to_dof_map = makeNodeToDofMap(part, probdef_ctwrapper);
        auto       dest_wgts       = std::vector< int >(mesh.getPartitions().size(), 0);
        for (auto node : part.getGhostNodes())
            if (const auto dof_it = node_to_dof_map.find(node); dof_it != node_to_dof_map.end())
                dest_wgts[node_to_part_map[node]] += dof_it->second.count();
        if (const int n_dests = std::ranges::count_if(dest_wgts, [](auto wgt) { return wgt > 0; }); n_dests > 0)
        {
            retval.sources.push_back(part_ind);
            retval.degrees.push_back(n_dests);
            for (int dest_ind = 0; auto dest_wgt : dest_wgts)
            {
                if (dest_wgt > 0)
                {
                    retval.destinations.push_back(dest_ind);
                    retval.weights.push_back(dest_wgt);
                }
                ++dest_ind;
            }
        }
        ++part_ind;
    }
    return retval;
}

template < detail::ProblemDef_c auto problem_def >
auto computeOptimalRankPermutation(const MpiComm&                comm,
                                   const Mesh&                   mesh,
                                   ConstexprValue< problem_def > probdef_ctwrapper) -> std::vector< int >
{
    L3STER_PROFILE_FUNCTION;
    if (comm.getSize() == 1)
        return {0};

    if (comm.getRank() == 0)
    {
        const auto [sources, degrees, destinations, weights] = makeCommVolumeInfo(mesh, probdef_ctwrapper);
        const int          optimal_rank = comm.distGraphCreate(sources, degrees, destinations, weights, true).getRank();
        std::vector< int > retval(comm.getSize());
        comm.gather(&optimal_rank, retval.data(), 1, 0);
        return retval;
    }
    else
    {
        const auto empty        = std::views::empty< int >;
        const int  optimal_rank = comm.distGraphCreate(empty, empty, empty, empty, true).getRank();
        int        _{};
        comm.gather(&optimal_rank, &_, 1, 0);
        return {};
    }
}

template < el_o_t order, MeshFormat mesh_format >
Mesh readAndConvertMesh(std::string_view mesh_file, MeshFormatTag< mesh_format > format_tag)
{
    auto mesh = readMesh(mesh_file, format_tag);
    mesh.getPartitions().front().initDualGraph();
    mesh.getPartitions().front() = convertMeshToOrder< order >(mesh.getPartitions()[0]);
    return mesh;
}
} // namespace detail::dist_mesh

template < detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
MeshPartition distributeMesh(const MpiComm&                comm,
                             const Mesh&                   mesh,
                             const std::vector< d_id_t >&  boundaries,
                             ConstexprValue< problem_def > probdef_ctwrpr = {})
{
    L3STER_PROFILE_FUNCTION;
    if (comm.getSize() == 1)
        return mesh.getPartitions().front();

    const auto node_throughputs = gatherNodeThroughputs(comm);
    if (comm.getRank() == 0)
    {
        auto          mesh_parted = partitionMesh(mesh, comm.getSize(), boundaries, node_throughputs, probdef_ctwrpr);
        const auto    permutation = detail::dist_mesh::computeOptimalRankPermutation(comm, mesh_parted, probdef_ctwrpr);
        MeshPartition my_partition;
        for (int unpermuted_rank = 0; MeshPartition & part : mesh_parted.getPartitions())
        {
            const auto dest_rank = permutation[unpermuted_rank];
            if (dest_rank == comm.getRank())
                my_partition = std::move(part);
            else
            {
                const auto serialized_part = SerializedPartition{part};
                sendPartition(comm, serialized_part, dest_rank);
            }
            ++unpermuted_rank;
        }
        return my_partition;
    }
    else
    {
        detail::dist_mesh::computeOptimalRankPermutation(comm, mesh, probdef_ctwrpr);
        return deserializePartition(receivePartition(comm, 0));
    }
}

template < el_o_t order, MeshFormat mesh_format, detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
MeshPartition readAndDistributeMesh(const MpiComm&               comm,
                                    std::string_view             mesh_file,
                                    MeshFormatTag< mesh_format > format_tag,
                                    const std::vector< d_id_t >& boundaries,
                                    ConstexprValue< order >                      = {},
                                    ConstexprValue< problem_def > probdef_ctwrpr = {})
{
    const Mesh mesh =
        comm.getRank() == 0 ? detail::dist_mesh::readAndConvertMesh< order >(mesh_file, format_tag) : Mesh{};
    return distributeMesh(comm, mesh, boundaries, probdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
