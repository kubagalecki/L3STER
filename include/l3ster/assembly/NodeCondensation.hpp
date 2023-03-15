#ifndef L3STER_NODECONDENSATION_HPP
#define L3STER_NODECONDENSATION_HPP

#include "l3ster/assembly/ProblemDefinition.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

namespace lstr::detail
{
template < ProblemDef_c auto problem_def >
auto getElementBoundaryNodes(const MeshPartition& mesh, ConstexprValue< problem_def >) -> std::vector< n_id_t >
{
    const auto problem_domains_view = problem_def | std::views::transform([](const auto& pair) { return pair.first; });
    std::vector< n_id_t > retval;
    retval.reserve(mesh.getAllNodes().size());
    mesh.visit(
        [&]< ElementTypes ET, el_o_t EO >(const Element< ET, EO >& element) {
            std::ranges::copy(getBoundaryNodes(element), std::back_inserter(retval));
        },
        problem_domains_view);
    util::sortRemoveDup(retval);
    return retval;
}

inline auto computeCondensedElementBoundaryNodeIds(const MpiComm&               comm,
                                                   const MeshPartition&         mesh,
                                                   const std::vector< n_id_t >& uncondensed_boundary_nodes)
    -> std::vector< n_id_t >
{
    const int my_rank   = comm.getRank();
    const int comm_size = comm.getSize();

    const size_t n_owned_boundary_nodes =
        std::ranges::count_if(uncondensed_boundary_nodes, [&](auto node) { return mesh.isOwnedNode(node); });
    const auto max_msg_size             = std::invoke([&] {
        size_t retval{};
        comm.allReduce(std::views::single(n_owned_boundary_nodes + 1u), &retval, MPI_MAX);
        return retval;
    });
    auto       owned_boundary_nodes_msg = std::invoke([&] {
        ArrayOwner< n_id_t > retval(max_msg_size);
        retval.front() = n_owned_boundary_nodes;
        std::ranges::copy_if(
            uncondensed_boundary_nodes, std::next(retval.begin()), [&](auto node) { return mesh.isOwnedNode(node); });
        return retval;
    });

    std::vector< n_id_t > retval(uncondensed_boundary_nodes.size());
    if (comm_size == 1)
    {
        std::iota(retval.begin(), retval.end(), 0ul);
        return retval;
    }
    ArrayOwner< n_id_t > proc_buf(max_msg_size), comm_buf(max_msg_size);
    const auto           process_received_boundary_ids = [&](int sender_rank) {
        auto&      buf_to_process = sender_rank == my_rank ? owned_boundary_nodes_msg : proc_buf;
        const auto off_rank_nodes = buf_to_process | std::views::drop(1) | std::views::take(buf_to_process.front());
        for (size_t i = 0; auto& cond_id : retval)
            cond_id += std::distance(off_rank_nodes.begin(),
                                     std::ranges::lower_bound(off_rank_nodes, uncondensed_boundary_nodes[i++]));
    };

    auto request = comm.broadcastAsync(my_rank == 0 ? owned_boundary_nodes_msg : comm_buf, 0);
    for (int sender_rank = 1; sender_rank < comm_size; ++sender_rank)
    {
        request.wait();
        std::swap(proc_buf, comm_buf);
        request = comm.broadcastAsync(my_rank == sender_rank ? owned_boundary_nodes_msg : comm_buf, sender_rank);
        process_received_boundary_ids(sender_rank - 1);
    }
    request.wait();
    std::swap(proc_buf, comm_buf);
    process_received_boundary_ids(comm_size - 1);
    return retval;
}

class NodeCondensationMap
{
public:
    NodeCondensationMap(const std::vector< n_id_t >& nodes_to_condense, std::vector< n_id_t > condensed_ids)
        : m_condensed_ids{std::move(condensed_ids)}
    {
        m_forward_map.reserve(m_condensed_ids.size());
        m_inverse_map.reserve(m_condensed_ids.size());
        for (size_t local_id = 0; auto uncond_node : nodes_to_condense)
        {
            const auto condensed_id = m_condensed_ids[local_id];
            const auto map_entry    = std::array{static_cast< n_id_t >(local_id), condensed_id};
            m_forward_map.emplace(uncond_node, map_entry);
            m_inverse_map.emplace(condensed_id, uncond_node);
            ++local_id;
        }
    }
    template < ProblemDef_c auto problem_def >
    static auto makeBoundaryNodeCondensationMap(const MpiComm&                comm,
                                                const MeshPartition&          mesh,
                                                ConstexprValue< problem_def > probdef_ctwrpr)
    {
        const auto boundary_nodes           = getElementBoundaryNodes(mesh, probdef_ctwrpr);
        auto       condensed_boundary_nodes = computeCondensedElementBoundaryNodeIds(comm, mesh, boundary_nodes);
        return NodeCondensationMap{boundary_nodes, condensed_boundary_nodes};
    }

    [[nodiscard]] auto mapToLocal(n_id_t id) const -> n_id_t { return m_forward_map.at(id).front(); }
    [[nodiscard]] auto mapToGlobal(n_id_t id) const -> n_id_t { return m_forward_map.at(id).back(); }
    [[nodiscard]] auto mapInverse(n_id_t id) const -> n_id_t { return m_inverse_map.at(id); }
    [[nodiscard]] auto size() const -> size_t { return m_forward_map.size(); }
    [[nodiscard]] auto getCondensedIds() const -> const std::vector< n_id_t >& { return m_condensed_ids; }

private:
    std::vector< n_id_t >                                             m_condensed_ids;
    robin_hood::unordered_flat_map< n_id_t, std::array< n_id_t, 2 > > m_forward_map;
    robin_hood::unordered_flat_map< n_id_t, n_id_t >                  m_inverse_map;
};

inline auto getCondensedOwnedNodesView(const MeshPartition& mesh, const NodeCondensationMap& cond_map)
{
    return cond_map.getCondensedIds() |
           std::views::filter([&](n_id_t node) { return mesh.isOwnedNode(cond_map.mapInverse(node)); });
}
} // namespace lstr::detail
#endif // L3STER_NODECONDENSATION_HPP
