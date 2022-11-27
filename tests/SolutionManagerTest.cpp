#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/assembly/AlgebraicSystemManager.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    const MpiComm    comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5., 6.};
    constexpr auto   mesh_order = 2;
    auto             mesh       = makeSquareMesh(node_dist);
    mesh.getPartitions()[0].initDualGraph();
    mesh.getPartitions()[0]    = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});

    // Define multiple problems to check correct indexing by SolutionManager
    constexpr auto problem_def1        = std::array{Pair{d_id_t{domain_id}, std::array{true, false}},
                                             Pair{d_id_t{bot_boundary}, std::array{true, true}}};
    constexpr auto problem_def_ctwrpr1 = ConstexprValue< problem_def1 >{};
    constexpr auto n_fields1           = detail::deduceNFields(problem_def1);
    constexpr auto field_inds1         = std::array< size_t, n_fields1 >{0, 2};
    auto           system_manager1     = AlgebraicSystemManager{comm, my_partition, problem_def_ctwrpr1};

    constexpr auto problem_def2        = std::array{Pair{d_id_t{domain_id}, std::array{true}}};
    constexpr auto problem_def_ctwrpr2 = ConstexprValue< problem_def2 >{};
    constexpr auto n_fields2           = detail::deduceNFields(problem_def2);
    constexpr auto field_inds2         = std::array< size_t, n_fields2 >{1};
    auto           system_manager2     = AlgebraicSystemManager{comm, my_partition, problem_def_ctwrpr2};

    auto solution_manager = SolutionManager{my_partition, comm, n_fields1 + n_fields2};

    // Set values for the first problem
    auto solution1 = system_manager1.makeSolutionMultiVector();
    solution1->modify_host();
    {
        auto solution_view = solution1->getDataNonConst(0);
        computeValuesAtNodes(my_partition,
                             std::views::single(domain_id),
                             system_manager1.getRhsMap(),
                             ConstexprValue< std::array{0} >{},
                             std::array{1.},
                             solution_view);
        computeValuesAtNodes(my_partition,
                             std::views::single(bot_boundary),
                             system_manager1.getRhsMap(),
                             ConstexprValue< std::array{1} >{},
                             std::array{2.},
                             solution_view);
    }
    solution1->sync_device();

    // Set values for the second problem
    auto solution2 = system_manager2.makeSolutionMultiVector();
    solution2->modify_host();
    {
        auto solution_view = solution2->getDataNonConst(0);
        computeValuesAtNodes(my_partition,
                             std::views::single(domain_id),
                             system_manager2.getRhsMap(),
                             ConstexprValue< std::array{0} >{},
                             std::array{3.},
                             solution_view);
    }
    solution2->sync_device();

    CHECK_THROWS(std::ignore = solution_manager.getNodalValues(0));

    // Update values in the solution manager
    {
        const auto solution_vector = solution1->getVector(0);
        solution_manager.updateSolution(
            my_partition, *solution_vector, system_manager1.getRhsMap(), field_inds1, problem_def_ctwrpr1);
    }
    {
        const auto solution_vector = solution2->getVector(0);
        solution_manager.updateSolution(
            my_partition, *solution_vector, system_manager2.getRhsMap(), field_inds2, problem_def_ctwrpr2);
    }
    solution_manager.communicateSharedValues();
    CHECK_THROWS(solution_manager.communicateSharedValues());

    bool success = true;
    // Check the first problem's fields
    {
        const auto field0_vals = solution_manager.getNodalValues(field_inds1[0]);
        const auto field1_vals = solution_manager.getNodalValues(field_inds1[1]);
        const auto n_nodes     = my_partition.getNodes().size() + my_partition.getGhostNodes().size();
        const auto n_rows      = static_cast< size_t >(field0_vals.size());
        if (n_rows != n_nodes)
        {
            std::stringstream err_msg;
            err_msg
                << "Error on rank " << comm.getRank()
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
            err_msg << "Error on rank " << comm.getRank() << ": Field 2 had " << bad_nodes.size() << " incorrect value"
                    << (bad_nodes.size() == 1 ? "" : "s") << " (!= 2.)\nThe affected nodes are:\n";
            const auto log_bad_node = [&](n_id_t node) {
                err_msg << node << ": ";
                if (my_partition.isGhostNode(node))
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
    }
    // Check the second problem's field
    {
        const auto field_vals = solution_manager.getNodalValues(field_inds2[0]);
        if (const auto non_threes = std::ranges::count_if(field_vals, [](auto v) { return v != 3.; }); non_threes != 0)
        {
            std::stringstream err_msg;
            err_msg << "Error on rank " << comm.getRank() << ": Field 1 had " << non_threes << " incorrect value"
                    << (non_threes == 1 ? "" : "s") << " (!= 3.)\n ";
            std::cerr << err_msg.str();
            success = false;
        }
    }
    if (success)
        return EXIT_SUCCESS;
    else
        return EXIT_FAILURE;
}
