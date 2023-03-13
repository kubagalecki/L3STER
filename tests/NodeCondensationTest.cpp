#include "l3ster/assembly/NodeCondensation.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    MpiComm          comm{MPI_COMM_WORLD};

    constexpr auto problem_def       = std::array{Pair{d_id_t{0}, std::array{true}}};
    constexpr auto problemdef_ctwrpr = ConstexprValue< problem_def >{};

    constexpr auto        mesh_order = 2;
    constexpr std::array  node_dist{0., 1., 2.};
    std::vector< d_id_t > boundaries(6);
    std::iota(boundaries.begin(), boundaries.end(), 1);
    const auto mesh = distributeMesh(comm,
                                     comm.getRank() == 0 ? std::invoke([&] {
                                         auto retval = makeCubeMesh(node_dist);
                                         retval.getPartitions().front().initDualGraph();
                                         retval.getPartitions().front() =
                                             convertMeshToOrder< mesh_order >(retval.getPartitions().front());
                                         return retval;
                                     })
                                                         : Mesh{},
                                     boundaries,
                                     problemdef_ctwrpr);

    const auto condensation_map =
        detail::NodeCondensationMap::makeBoundaryNodeCondensationMap(comm, mesh, problemdef_ctwrpr);
    const auto global_nodes_to_condense = detail::getElementBoundaryNodes(mesh, problemdef_ctwrpr);
    REQUIRE(condensation_map.size() == global_nodes_to_condense.size());
    for (auto n : global_nodes_to_condense)
        REQUIRE(condensation_map.mapToLocal(n) < global_nodes_to_condense.size());
    std::vector< n_id_t > uncondensed_owned, condensed_ghost, condensed_owned;
    for (auto n : global_nodes_to_condense)
    {
        if (mesh.isOwnedNode(n))
        {
            uncondensed_owned.push_back(n);
            condensed_owned.push_back(condensation_map.mapToGlobal(n));
        }
        else
            condensed_ghost.push_back(condensation_map.mapToGlobal(n));
    }

    const auto gather = [&](std::vector< n_id_t > nodes) {
        size_t max_len{};
        comm.allReduce(std::views::single(nodes.size()), &max_len, MPI_MAX);
        nodes.resize(max_len, std::numeric_limits< n_id_t >::max());
        auto gather_buf = comm.getRank() == 0
                            ? std::vector< n_id_t >(max_len * comm.getSize(), std::numeric_limits< n_id_t >::max())
                            : std::vector< n_id_t >{};
        comm.gather(nodes, gather_buf.begin(), 0);
        return gather_buf;
    };

    // Union of all ranks' owned condensed nodes forms an interval of length equal to the number of all condensed nodes
    const auto gathered_owned_condensed = gather(condensed_owned);
    if (comm.getRank() == 0)
    {
        auto nodes = gathered_owned_condensed;
        util::sortRemoveDup(nodes);
        if (nodes.size() > 0 and nodes.back() == std::numeric_limits< n_id_t >::max())
            nodes.pop_back();

        const auto elems_per_edge          = (node_dist.size() - 1);
        const auto n_elems                 = elems_per_edge * elems_per_edge * elems_per_edge;
        const auto internal_nodes_per_elem = (mesh_order - 1) * (mesh_order - 1) * (mesh_order - 1);
        const auto n_internal_nodes        = internal_nodes_per_elem * n_elems;
        const auto nodes_per_edge          = elems_per_edge * mesh_order + 1;
        const auto n_all_nodes             = nodes_per_edge * nodes_per_edge * nodes_per_edge;
        REQUIRE(nodes.size() == n_all_nodes - n_internal_nodes);
        REQUIRE(std::ranges::equal(nodes, std::views::iota(0u, nodes.size())));
    }

    // Ghost condensed nodes have the correct ID
    const auto gathered_owned_uncondensed = gather(std::move(uncondensed_owned));
    const auto gathered_ghost_condensed   = gather(std::move(condensed_ghost));
    const auto gathered_ghost_uncondensed =
        gather(std::vector< n_id_t >{mesh.getGhostNodes().begin(), mesh.getGhostNodes().end()});
    if (comm.getRank() == 0)
    {
        for (size_t i = 0; auto ghost_uncond : gathered_ghost_uncondensed)
        {
            if (ghost_uncond == std::numeric_limits< n_id_t >::max())
            {
                ++i;
                continue;
            }

            const auto owned_ind = std::distance(gathered_owned_uncondensed.begin(),
                                                 std::ranges::find(gathered_owned_uncondensed, ghost_uncond));
            REQUIRE(gathered_ghost_condensed[i++] == gathered_owned_condensed[owned_ind]);
        }
    }
}
