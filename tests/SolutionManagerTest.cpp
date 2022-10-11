#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/assembly/AlgebraicSystemManager.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "Common.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    const MpiComm comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    constexpr auto   mesh_order = 2;
    auto             mesh       = makeSquareMesh(node_dist);
    mesh.getPartitions()[0].initDualGraph();
    mesh.getPartitions()[0]    = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});

    constexpr auto problem_def        = std::array{Pair{d_id_t{domain_id}, std::array{true, false}},
                                            Pair{d_id_t{bot_boundary}, std::array{true, true}}};
    constexpr auto problem_def_ctwrpr = ConstexprValue< problem_def >{};

    constexpr auto n_fields         = detail::deduceNFields(problem_def);
    auto           system_manager   = AlgebraicSystemManager{comm, my_partition, problem_def_ctwrpr};
    auto           solution_manager = SolutionManager{my_partition, comm, n_fields};
    constexpr auto field_inds       = std::views::iota(0u, n_fields);
    auto           solution         = system_manager.makeSolutionMultiVector();

    {
        auto solution_vector = solution->getVectorNonConst(0);
        computeValuesAtNodes(
            [](const auto&) {
                Eigen::Matrix< val_t, 1, 1 > retval;
                retval[0] = 1.;
                return retval;
            },
            my_partition,
            std::views::single(domain_id),
            system_manager.getRhsMap(),
            ConstexprValue< std::array{0} >{},
            *solution_vector);
        computeValuesAtNodes(
            [](const auto&) {
                Eigen::Matrix< val_t, 1, 1 > retval;
                retval[0] = 2.;
                return retval;
            },
            my_partition,
            std::views::single(bot_boundary),
            system_manager.getRhsMap(),
            ConstexprValue< std::array{1} >{},
            *solution_vector);
    }

    CHECK_THROWS(std::ignore = solution_manager.getNodalValues(0))
    {
        const auto solution_vector = solution->getVector(0);
        solution_manager.updateSolution(
            my_partition, *solution_vector, system_manager.getRhsMap(), field_inds, problem_def_ctwrpr);
        solution_manager.updateSolution(
            my_partition, *solution_vector, system_manager.getRhsMap(), field_inds, problem_def_ctwrpr);
        solution_manager.communicateSharedValues();
        CHECK_THROWS(solution_manager.communicateSharedValues())
    }

    const auto field0_vals = solution_manager.getNodalValues(0);
    const auto field1_vals = solution_manager.getNodalValues(1);

    bool       success = true;
    const auto n_nodes = my_partition.getNodes().size() + my_partition.getGhostNodes().size();
    const auto n_rows  = static_cast< size_t >(field0_vals.size());
    if (n_rows != n_nodes)
    {
        std::stringstream err_msg;
        err_msg << "Error on rank " << comm.getRank()
                << ": the nodal solution multivector has a different number of rows than the number of nodes in the "
                   "mesh partition\nNumber of nodes in the partition: "
                << n_nodes << "\nNumber of rows: " << n_rows << '\n';
        std::cerr << err_msg.str();
        success = false;
    }
    if (const auto non_ones = std::ranges::count_if(field0_vals, [](auto v) { return v != 1.; }); non_ones != 0)
    {
        std::stringstream err_msg;
        err_msg << "Error on rank " << comm.getRank() << ": Field 0 had " << non_ones << " incorrect value"
                << (non_ones == 1 ? "" : "s") << " (!= 1.)\n ";
        std::cerr << err_msg.str();
        success = false;
    }

    std::vector< n_id_t > bad_nodes;
    const auto            lookup_node_local_ind = [&](n_id_t node) {
        const auto owned_it = std::ranges::find(my_partition.getNodes(), node);
        if (owned_it != my_partition.getNodes().end())
            return std::distance(my_partition.getNodes().begin(), owned_it);
        else
            return std::distance(my_partition.getGhostNodes().begin(),
                                 std::ranges::find(my_partition.getGhostNodes(), node));
    };
    my_partition.visit(
        [&](const auto& element) {
            for (auto node : element.getNodes())
                if (field1_vals[lookup_node_local_ind(node)] != 2.)
                    bad_nodes.push_back(node);
        },
        std::views::single(bot_boundary));
    if (bad_nodes.size() != 0)
    {
        std::ranges::sort(bad_nodes);
        const auto erase_range = std::ranges::unique(bad_nodes);
        bad_nodes.erase(std::ranges::begin(erase_range), std::ranges::end(erase_range));
        std::stringstream err_msg;
        err_msg << "Error on rank " << comm.getRank() << ": Field 1 had " << bad_nodes.size() << " incorrect value"
                << (bad_nodes.size() == 1 ? "" : "s") << " (!= 2.)\nThe affected nodes are:\n";
        const auto log_bad_node = [&](n_id_t node) {
            err_msg << node << ": ";
            if (std::ranges::binary_search(my_partition.getGhostNodes(), node))
                err_msg << "ghost\n";
            else
                err_msg << "owned\n";
        };
        for (auto n : bad_nodes)
            log_bad_node(n);
        err_msg << '\n';
        std::cerr << err_msg.str();
        success = false;
    }
    if (success)
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}
