#ifndef L3STER_COMM_DISTRIBUTEMESH_HPP
#define L3STER_COMM_DISTRIBUTEMESH_HPP

#include "l3ster/comm/GatherNodeThroughputs.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/PartitionMesh.hpp"
#include "l3ster/mesh/ReadMesh.hpp"

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
template < el_o_t... orders, ProblemDef problem_def >
auto makeGhostToWgtMap(const mesh::MeshPartition< orders... >& mesh, util::ConstexprValue< problem_def > probdef_ctwrpr)
    -> robin_hood::unordered_flat_map< n_id_t, int >
{
    constexpr auto n_fields = problem_def.n_fields;
    auto           retval   = robin_hood::unordered_flat_map< n_id_t, int >(mesh.getGhostNodes().size());
    if constexpr (n_fields == 0)
        for (auto n : mesh.getGhostNodes())
            retval[n] = 1;
    else
    {
        const auto is_ghost_node = util::negatePredicate(mesh.getOwnedNodePredicate());
        auto dof_map = robin_hood::unordered_flat_map< n_id_t, std::bitset< problem_def.n_fields > >{retval.size()};
        const auto visit_dom = [&]< DomainDef< n_fields > dom_def >(util::ConstexprValue< dom_def >) {
            const auto dom_bitset = util::toStdBitset(dom_def.active_fields);
            const auto visit_elem = [&](const auto& element) {
                for (auto node : element.getNodes() | std::views::filter(is_ghost_node))
                    dof_map[node] |= dom_bitset;
            };
            mesh.visit(visit_elem, dom_def.domain);
        };
        forConstexpr(visit_dom, probdef_ctwrpr);
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
        [](const mesh::MeshPartition< orders... >& part) { return part.getOwnedNodes().size(); });
    return retval;
}

struct CommVolumeInfo
{
    std::vector< int > sources, degrees, destinations, weights;
};
template < el_o_t... orders, ProblemDef problem_def >
auto makeCommVolumeInfo(const util::ArrayOwner< mesh::MeshPartition< orders... > >& mesh_parts,
                        util::ConstexprValue< problem_def >                         probdef_ctwrapper) -> CommVolumeInfo
{
    const auto get_node_owner = [node_dist = makeNodeDist(mesh_parts)](n_id_t node) {
        const auto lb_iter = std::ranges::upper_bound(node_dist, node);
        return std::distance(node_dist.begin(), lb_iter);
    };
    auto retval    = CommVolumeInfo{};
    auto dest_wgts = util::ArrayOwner< int >(mesh_parts.size(), 0);
    for (auto&& [part_ind, part] : mesh_parts | std::views::enumerate)
    {
        const auto ghost_to_wgt_map = makeGhostToWgtMap(part, probdef_ctwrapper);
        std::ranges::fill(dest_wgts, 0);
        for (auto node : part.getGhostNodes())
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

template < el_o_t... orders, ProblemDef problem_def >
auto computeOptimizedRankPermutation(const MpiComm&                                              comm,
                                     const util::ArrayOwner< mesh::MeshPartition< orders... > >& mesh_parts,
                                     util::ConstexprValue< problem_def > probdef_ctwrapper) -> util::ArrayOwner< int >
{
    const auto& [sources, degrees, dests, wgts] = makeCommVolumeInfo(mesh_parts, probdef_ctwrapper);
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
    auto retval = util::ArrayOwner< int >(static_cast< size_t >(comm.getSize()));
    comm.gather(std::views::single(new_rank), retval.begin(), 0);
    return retval;
}

inline auto computeDefaultRankPermutation(const MpiComm& comm) -> util::ArrayOwner< int >
{
    auto retval = util::ArrayOwner< int >(static_cast< size_t >(comm.getSize()));
    std::iota(retval.begin(), retval.end(), 0);
    return retval;
}
} // namespace detail

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

template < el_o_t... orders >
auto receiveMesh(const MpiComm& comm,
                 int            src_rank,
                 util::TypePack< mesh::MeshPartition< orders... > >) -> mesh::MeshPartition< orders... >
{
    return receiveMesh< orders... >(comm, src_rank);
}

template < typename MeshGenerator, ProblemDef problem_def = EmptyProblemDef{} >
auto distributeMesh(const MpiComm&                       comm,
                    MeshGenerator&&                      mesh_generator,
                    util::ConstexprValue< problem_def >  probdef_ctwrpr = {},
                    [[maybe_unused]] const MeshDistOpts& opts           = {})
{
    L3STER_PROFILE_FUNCTION;
    using partition_t = std::remove_cvref_t< decltype(std::invoke(mesh_generator)) >;
    if (comm.getSize() == 1)
        return std::make_shared< partition_t >(std::invoke(mesh_generator));

    const auto node_throughputs = comm::gatherNodeThroughputs(comm);
    const auto mesh             = comm.getRank() == 0 ? std::invoke(mesh_generator) : partition_t{};
    auto       mesh_parted = comm.getRank() == 0 ? partitionMesh(mesh, comm.getSize(), node_throughputs, probdef_ctwrpr)
                                                 : util::ArrayOwner< partition_t >{};
    const auto permutation = opts.optimize ? detail::computeOptimizedRankPermutation(comm, mesh_parted, probdef_ctwrpr)
                                           : detail::computeDefaultRankPermutation(comm);
    if (comm.getRank() == 0)
    {
        auto reqs = std::vector< MpiComm::Request >{};
        reqs.reserve(mesh_parted.size() * 4);
        auto my_partition = partition_t{};
        for (auto&& [unpermuted_rank, part] : mesh_parted | std::views::enumerate)
        {
            const auto dest_rank = permutation.at(unpermuted_rank);
            if (dest_rank == 0)
                my_partition = std::move(part);
            else
                comm::sendMesh(comm, part, dest_rank, std::back_inserter(reqs));
        }
        MpiComm::Request::waitAll(reqs);
        return std::make_shared< partition_t >(std::move(my_partition));
    }
    else
        return std::make_shared< partition_t >(comm::receiveMesh(comm, 0, util::TypePack< partition_t >{}));
}
} // namespace comm

template < el_o_t                                     order,
           GeneratorFor_c< mesh::MeshPartition< 1 > > Generator,
           ProblemDef                                 problem_def = EmptyProblemDef{} >
auto generateAndDistributeMesh(const MpiComm&                      comm,
                               Generator&&                         mesh_generator,
                               util::ConstexprValue< order >       order_ctwrpr   = {},
                               util::ConstexprValue< problem_def > probdef_ctwrpr = {},
                               const MeshDistOpts& opts = {}) -> std::shared_ptr< mesh::MeshPartition< order > >
{
    const auto converted_mesh_generator = [&] {
        auto generated_mesh = std::invoke(std::forward< Generator >(mesh_generator));
        if constexpr (order_ctwrpr.value == 1)
            return generated_mesh;
        else
            return convertMeshToOrder< order >(generated_mesh);
    };
    return comm::distributeMesh(comm, converted_mesh_generator, probdef_ctwrpr, opts);
}

template < el_o_t order, mesh::MeshFormat mesh_format, ProblemDef problem_def = EmptyProblemDef{} >
auto readAndDistributeMesh(const MpiComm&                      comm,
                           std::string_view                    mesh_file,
                           mesh::MeshFormatTag< mesh_format >  format_tag,
                           const util::ArrayOwner< d_id_t >&   boundaries,
                           util::ConstexprValue< order >       order_ctwrpr   = {},
                           util::ConstexprValue< problem_def > probdef_ctwrpr = {},
                           const MeshDistOpts& opts = {}) -> std::shared_ptr< mesh::MeshPartition< order > >
{
    const auto read_generator = [&] {
        return readMesh(mesh_file, boundaries, format_tag);
    };
    return generateAndDistributeMesh(comm, read_generator, order_ctwrpr, probdef_ctwrpr, opts);
}
} // namespace lstr
#endif // L3STER_COMM_DISTRIBUTEMESH_HPP
