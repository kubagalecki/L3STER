#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "DenseGraph.hpp"
#include "TestDataPath.h"

#include "Common.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};
    const MpiComm    comm{MPI_COMM_WORLD};

    const std::array node_dist{0., 1., 2., 3., 4., 5.};
    const auto       mesh         = makeCubeMesh(node_dist);
    const auto       my_partition = distributeMesh(comm, mesh, {});
    const auto&      full_mesh    = mesh.getPartitions().front();

    constexpr auto problem_def           = std::array{Pair{d_id_t{0}, std::array{false, true}},
                                            Pair{d_id_t{1}, std::array{true, false}},
                                            Pair{d_id_t{3}, std::array{true, true}}};
    constexpr auto problem_def_ctwrapper = ConstexprValue< problem_def >{};
    const auto     dof_intervals         = computeDofIntervals(my_partition, problem_def_ctwrapper, comm);
    const auto     global_dof_map        = NodeToGlobalDofMap{my_partition, dof_intervals};
    const auto sparsity_graph = detail::makeSparsityGraph(my_partition, global_dof_map, problem_def_ctwrapper, comm);

    const auto all_dofs    = detail::getNodeDofs(full_mesh.getAllNodes(), dof_intervals);
    const auto dense_graph = DenseGraph{full_mesh, problem_def_ctwrapper, dof_intervals, all_dofs.size()};

    REQUIRE(sparsity_graph->getGlobalNumRows() == all_dofs.size());
    REQUIRE(sparsity_graph->getGlobalNumCols() == all_dofs.size());

    const auto n_my_rows = sparsity_graph->getLocalNumRows();
    auto       view =
        tpetra_crsgraph_t::nonconst_global_inds_host_view_type("Global cols", sparsity_graph->getGlobalNumCols());
    size_t row_size = 0;

    const auto check_row_global_cols = [&](const auto& dense_row) {
        REQUIRE(row_size == dense_row.count());
        REQUIRE(std::ranges::none_of(std::views::counted(view.data(), row_size),
                                     [&](global_dof_t col_ind) { return not dense_row.test(col_ind); }));
    };

    for (size_t local_row = 0; local_row < n_my_rows; ++local_row)
    {
        const auto global_row_ind = sparsity_graph->getRowMap()->getGlobalElement(local_row);
        sparsity_graph->getGlobalRowCopy(global_row_ind, view, row_size);
        check_row_global_cols(dense_graph.getRow(global_row_ind));
    }
}
