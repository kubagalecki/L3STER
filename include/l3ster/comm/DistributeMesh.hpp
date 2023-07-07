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
template < el_o_t... orders, detail::ProblemDef_c auto problem_def >
auto makeNodeToDofMap(const mesh::MeshPartition< orders... >& mesh,
                      util::ConstexprValue< problem_def >     probdef_ctwrapper)
    -> robin_hood::unordered_flat_map< n_id_t, std::bitset< deduceNFields(problem_def) > >
{
    auto retval =
        robin_hood::unordered_flat_map< n_id_t, std::bitset< deduceNFields(problem_def) > >{mesh.getAllNodes().size()};
    forConstexpr(
        [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
            constexpr auto dom_id   = dom_def.first;
            constexpr auto dom_dofs = util::getTrueInds< dom_def.second >();
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

template < el_o_t... orders >
auto makeNodeToPartMap(const std::vector< mesh::MeshPartition< orders... > >& mesh_parts) -> std::vector< int >
{
    const size_t n_nodes =
        std::ranges::max(mesh_parts | std::views::filter([](const mesh::MeshPartition< orders... >& part) {
                             return not part.getOwnedNodes().empty();
                         }) |
                         std::views::transform(
                             [](const mesh::MeshPartition< orders... >& part) { return part.getOwnedNodes().back(); }));
    std::vector< int > retval(n_nodes);
    for (int part_ind = 0; const mesh::MeshPartition< orders... >& part : mesh_parts)
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
template < el_o_t... orders, detail::ProblemDef_c auto problem_def >
auto makeCommVolumeInfo(const std::vector< mesh::MeshPartition< orders... > >& mesh_parts,
                        util::ConstexprValue< problem_def >                    probdef_ctwrapper)
{
    const auto node_to_part_map = makeNodeToPartMap(mesh_parts);
    auto       retval           = CommVolumeInfo{};
    for (int part_ind = 0; const mesh::MeshPartition< orders... >& part : mesh_parts)
    {
        const auto node_to_dof_map = makeNodeToDofMap(part, probdef_ctwrapper);
        auto       dest_wgts       = std::vector< int >(mesh_parts.size());
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

template < el_o_t... orders, detail::ProblemDef_c auto problem_def >
auto computeOptimalRankPermutation(const MpiComm&                                         comm,
                                   const std::vector< mesh::MeshPartition< orders... > >& mesh,
                                   util::ConstexprValue< problem_def > probdef_ctwrapper) -> std::vector< int >
{
    L3STER_PROFILE_FUNCTION;
    if (comm.getSize() == 1)
        return {0};

    if (comm.getRank() == 0)
    {
        const auto [sources, degrees, destinations, weights] = makeCommVolumeInfo(mesh, probdef_ctwrapper);
        const int          optimal_rank = comm.distGraphCreate(sources, degrees, destinations, weights, true).getRank();
        std::vector< int > retval(comm.getSize());
        comm.gather(std::views::single(optimal_rank), retval.begin(), 0);
        return retval;
    }
    else
    {
        const auto empty        = std::views::empty< int >;
        const int  optimal_rank = comm.distGraphCreate(empty, empty, empty, empty, true).getRank();
        comm.gather(std::views::single(optimal_rank), &std::ignore, 0);
        return {};
    }
}

inline auto computeDefaultRankPermutation(const MpiComm& comm)
{
    std::vector< int > retval(comm.getSize());
    std::iota(begin(retval), end(retval), 0);
    return retval;
}

template < el_o_t order, mesh::MeshFormat mesh_format >
auto readAndConvertMesh(std::string_view mesh_file, mesh::MeshFormatTag< mesh_format > format_tag)
    -> mesh::MeshPartition< order >
{
    auto mesh = readMesh(mesh_file, format_tag);
    mesh.initDualGraph();
    return convertMeshToOrder< order >(mesh);
}
} // namespace detail::dist_mesh

template < el_o_t... orders, detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
auto distributeMesh(const MpiComm&                          comm,
                    const mesh::MeshPartition< orders... >& mesh,
                    const std::vector< d_id_t >&            boundaries,
                    util::ConstexprValue< problem_def >     probdef_ctwrpr = {}) -> mesh::MeshPartition< orders... >
{
    L3STER_PROFILE_FUNCTION;
    if (comm.getSize() == 1)
        return mesh;

    const auto node_throughputs = gatherNodeThroughputs(comm);
    auto       mesh_parted      = comm.getRank() == 0
                                    ? partitionMesh(mesh, comm.getSize(), boundaries, node_throughputs, probdef_ctwrpr)
                                    : std::vector< mesh::MeshPartition< orders... > >{};
    const auto permutation      = detail::dist_mesh::computeDefaultRankPermutation(comm);
    // const auto permutation = detail::dist_mesh::computeOptimalRankPermutation(comm, mesh_parted, probdef_ctwrpr);
    if (comm.getRank() == 0)
    {
        mesh::MeshPartition< orders... > my_partition;
        for (size_t unpermuted_rank = 0; mesh::MeshPartition< orders... > & part : mesh_parted)
        {
            const auto dest_rank = permutation.at(unpermuted_rank);
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
        return deserializePartition< orders... >(receivePartition(comm, 0));
}

template < el_o_t order, detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
auto generateAndDistributeMesh(const MpiComm&               comm,
                               auto&&                       mesh_generator,
                               const std::vector< d_id_t >& boundaries,
                               util::ConstexprValue< order >                      = {},
                               util::ConstexprValue< problem_def > probdef_ctwrpr = {}) -> mesh::MeshPartition< order >
    requires std::is_invocable_r_v< mesh::MeshPartition< 1 >, decltype(mesh_generator) >
{
    const auto mesh = comm.getRank() == 0 ? std::invoke([&] {
        auto gen_mesh = std::invoke(mesh_generator);
        gen_mesh.initDualGraph();
        return convertMeshToOrder< order >(gen_mesh);
    })
                                          : mesh::MeshPartition< order >{};
    return distributeMesh(comm, mesh, boundaries, probdef_ctwrpr);
}

template < el_o_t                    order,
           mesh::MeshFormat          mesh_format,
           detail::ProblemDef_c auto problem_def = detail::empty_problem_def_t{} >
auto readAndDistributeMesh(const MpiComm&                     comm,
                           std::string_view                   mesh_file,
                           mesh::MeshFormatTag< mesh_format > format_tag,
                           const std::vector< d_id_t >&       boundaries,
                           util::ConstexprValue< order >                      = {},
                           util::ConstexprValue< problem_def > probdef_ctwrpr = {}) -> mesh::MeshPartition< order >
{
    const auto mesh = comm.getRank() == 0 ? detail::dist_mesh::readAndConvertMesh< order >(mesh_file, format_tag)
                                          : mesh::MeshPartition< order >{};
    return distributeMesh(comm, mesh, boundaries, probdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
