#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

#include <iterator>

namespace lstr
{
namespace comm
{
template < el_o_t... orders, ProblemDef problem_def >
auto makeNodeToDofMap(const mesh::MeshPartition< orders... >& mesh, util::ConstexprValue< problem_def > probdef_ctwrpr)
    -> robin_hood::unordered_flat_map< n_id_t, std::bitset< problem_def.n_fields > >
{
    constexpr auto n_fields = problem_def.n_fields;
    using retval_t          = robin_hood::unordered_flat_map< n_id_t, std::bitset< problem_def.n_fields > >;
    auto retval             = retval_t{mesh.getAllNodes().size()};
    forConstexpr(
        [&]< DomainDef< n_fields > dom_def >(util::ConstexprValue< dom_def >) {
            constexpr auto dom_dofs = util::getTrueInds< dom_def.active_fields >();
            mesh.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                    {
                        auto& node_map_entry = retval[node];
                        for (auto dof : dom_dofs)
                            node_map_entry.set(dof);
                    }
                },
                dom_def.domain);
        },
        probdef_ctwrpr);
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
template < el_o_t... orders, ProblemDef problem_def >
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

inline auto computeDefaultRankPermutation(const MpiComm& comm)
{
    std::vector< int > retval(comm.getSize());
    std::iota(begin(retval), end(retval), 0);
    return retval;
}

template < el_o_t order, mesh::MeshFormat mesh_format >
auto readAndConvertMesh(std::string_view                   mesh_file,
                        const util::ArrayOwner< d_id_t >&  boundary_ids,
                        mesh::MeshFormatTag< mesh_format > format_tag) -> mesh::MeshPartition< order >
{
    auto mesh = readMesh(mesh_file, boundary_ids, format_tag);
    return convertMeshToOrder< order >(mesh);
}

template < el_o_t... orders >
void sendMesh(const MpiComm&                                comm,
              const mesh::MeshPartition< orders... >&       mesh,
              int                                           dest_rank,
              std::output_iterator< MpiComm::Request > auto reqs_out)
{
    const auto num_domains  = mesh.getNDomains();
    const auto boundary_ids = mesh.getBoundaryIdsCopy();
    auto       descr_ints   = util::ArrayOwner< size_t >(num_domains + boundary_ids.size() + 2);
    descr_ints[0]           = mesh.getOwnedNodes().size();
    descr_ints[1]           = num_domains;
    const auto write_iter   = std::ranges::copy(mesh.getDomainIds(), std::next(descr_ints.begin(), 2)).out;
    std::ranges::copy(boundary_ids, write_iter);
    int        tag         = 0;
    const auto local_req   = comm.sendAsync(descr_ints, dest_rank, tag++); // wait via dtor
    *reqs_out++            = comm.sendAsync(mesh.getAllNodes(), dest_rank, tag++);
    const auto send_domain = [&](const mesh::Domain< orders... >& domain) {
        const auto send_elvec = [&](const auto& el_vec) {
            *reqs_out++ = comm.sendAsync(el_vec, dest_rank, tag++);
        };
        domain.elements.visitVectors(send_elvec);
        *reqs_out++ = comm.sendAsync(std::span{&domain.dim, 1}, dest_rank, tag++);
    };
    for (d_id_t domain_id : mesh.getDomainIds())
        send_domain(mesh.getDomain(domain_id));
}

template < el_o_t... orders >
auto receiveMesh(const MpiComm& comm, int src_rank) -> mesh::MeshPartition< orders... >
{
    constexpr size_t num_elem_types    = mesh::Domain< orders... >::el_univec_t::num_types;
    const auto       descr_ints_status = comm.probe(src_rank, 0);
    const auto       descr_sz          = descr_ints_status.numElems< size_t >();
    util::throwingAssert(descr_sz > 1, "Invalid mesh description integer vector");
    auto descr_ints = util::ArrayOwner< size_t >(descr_sz);
    comm.receive(descr_ints, src_rank, 0);
    const auto num_owned_nodes = descr_ints[0];
    const auto num_domains     = descr_ints[1];

    int  tag  = 1;
    auto reqs = std::vector< MpiComm::Request >{};
    reqs.reserve(num_domains * (num_elem_types + 1) + 1);

    const size_t num_nodes = comm.probe(src_rank, tag).numElems< n_id_t >();
    auto         nodes     = util::ArrayOwner< n_id_t >(num_nodes);
    reqs.push_back(comm.receiveAsync(nodes, src_rank, tag++));
    auto domain_map = typename mesh::MeshPartition< orders... >::domain_map_t{};

    for (auto domain_id : descr_ints | std::views::drop(2) | std::views::take(num_domains))
    {
        auto& domain = domain_map[static_cast< d_id_t >(domain_id)];
        domain.elements.visitVectors([&]< typename Element >(std::vector< Element >& el_vec) {
            const size_t sz = comm.probe(src_rank, tag).numElems< Element >();
            el_vec.resize(sz);
            reqs.push_back(comm.receiveAsync(el_vec, src_rank, tag++));
        });
        reqs.push_back(comm.receiveAsync(std::span{&domain.dim, 1}, src_rank, tag++));
    }
    const auto boundary_ids = util::ArrayOwner< d_id_t >{descr_ints | std::views::drop(num_domains + 2)};
    MpiComm::Request::waitAll(reqs);
    return {std::move(domain_map), std::move(nodes), num_owned_nodes, boundary_ids};
}

template < el_o_t... orders, ProblemDef problem_def = EmptyProblemDef{} >
auto distributeMesh(const MpiComm&                      comm,
                    mesh::MeshPartition< orders... >&&  mesh,
                    util::ConstexprValue< problem_def > probdef_ctwrpr = {},
                    mesh::PartitioningOpts opts = {}) -> std::shared_ptr< mesh::MeshPartition< orders... > >
{
    L3STER_PROFILE_FUNCTION;
    if (comm.getSize() == 1)
        return std::make_shared< mesh::MeshPartition< orders... > >(std::move(mesh));

    const auto node_throughputs = comm::gatherNodeThroughputs(comm);
    auto mesh_parted = comm.getRank() == 0 ? partitionMesh(mesh, comm.getSize(), node_throughputs, probdef_ctwrpr, opts)
                                           : util::ArrayOwner< mesh::MeshPartition< orders... > >{};
    const auto permutation = comm::computeDefaultRankPermutation(comm);
    // const auto permutation = detail::dist_mesh::computeOptimalRankPermutation(comm, mesh_parted, probdef_ctwrpr);
    if (comm.getRank() == 0)
    {
        auto reqs         = std::vector< MpiComm::Request >{};
        auto my_partition = mesh::MeshPartition< orders... >{};
        for (size_t unpermuted_rank = 0; mesh::MeshPartition< orders... > & part : mesh_parted)
        {
            const auto dest_rank = permutation.at(unpermuted_rank);
            if (dest_rank == comm.getRank())
                my_partition = std::move(part);
            else
                comm::sendMesh(comm, part, dest_rank, std::back_inserter(reqs));
            ++unpermuted_rank;
        }
        MpiComm::Request::waitAll(reqs);
        return std::make_shared< mesh::MeshPartition< orders... > >(std::move(my_partition));
    }
    else
        return std::make_shared< mesh::MeshPartition< orders... > >(comm::receiveMesh< orders... >(comm, 0));
}
} // namespace comm

template < el_o_t                                     order,
           GeneratorFor_c< mesh::MeshPartition< 1 > > Generator,
           ProblemDef                                 problem_def = EmptyProblemDef{} >
auto generateAndDistributeMesh(const MpiComm& comm,
                               Generator&&    mesh_generator,
                               util::ConstexprValue< order >                      = {},
                               util::ConstexprValue< problem_def > probdef_ctwrpr = {})
    -> std::shared_ptr< mesh::MeshPartition< order > >
{
    return comm::distributeMesh(comm,
                                comm.getRank() == 0 ? std::invoke([&] {
                                    auto generated_mesh = std::invoke(std::forward< Generator >(mesh_generator));
                                    if constexpr (order == 1)
                                        return generated_mesh;
                                    else
                                        return convertMeshToOrder< order >(generated_mesh);
                                })
                                                    : mesh::MeshPartition< order >{},
                                probdef_ctwrpr);
}

template < el_o_t order, mesh::MeshFormat mesh_format, ProblemDef problem_def = EmptyProblemDef{} >
auto readAndDistributeMesh(const MpiComm&                     comm,
                           std::string_view                   mesh_file,
                           mesh::MeshFormatTag< mesh_format > format_tag,
                           const util::ArrayOwner< d_id_t >&  boundaries,
                           util::ConstexprValue< order >                      = {},
                           util::ConstexprValue< problem_def > probdef_ctwrpr = {})
    -> std::shared_ptr< mesh::MeshPartition< order > >
{
    return comm::distributeMesh(comm,
                                comm.getRank() == 0
                                    ? comm::readAndConvertMesh< order >(mesh_file, boundaries, format_tag)
                                    : mesh::MeshPartition< order >{},
                                probdef_ctwrpr);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
