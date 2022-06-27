#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "DenseGraph.hpp"
#include "TestDataPath.h"

int main(int argc, char* argv[])
{
    using namespace lstr;

    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    const MpiComm comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5.};
    const auto       mesh         = makeCubeMesh(node_dist);
    auto             my_partition = distributeMesh(comm, mesh, {});

    constexpr auto problem_def    = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{false, true}},
                                                            Pair{d_id_t{1}, std::array{true, false}},
                                                            Pair{d_id_t{3}, std::array{true, true}}} >{};
    const auto     dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto     all_dofs       = detail::getNodeDofs(mesh.getPartitions()[0].getNodes(), dof_intervals);
    const auto     sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);
    const auto     dense_graph    = DenseGraph{mesh.getPartitions()[0], problem_def, dof_intervals, all_dofs};

    try
    {
        if (sparsity_graph->getGlobalNumRows() != all_dofs.size())
            throw std::logic_error{"Incorrect number of global rows"};
        if (sparsity_graph->getGlobalNumCols() != all_dofs.size())
            throw std::logic_error{"Incorrect number of global columns"};

        const auto                               n_my_rows = sparsity_graph->getNodeNumRows();
        std::vector< global_dof_t >              alloc(sparsity_graph->getGlobalNumCols());
        const Teuchos::ArrayView< global_dof_t > view{alloc};
        size_t                                   row_size = 0;

        const auto check_row_global_cols = [&](const auto& dense_row) {
            if (row_size != dense_row.count())
                throw std::logic_error{"Incorrect row size"};
            if (std::ranges::any_of(alloc | std::views::take(row_size),
                                    [&](global_dof_t col_ind) { return not dense_row.test(col_ind); }))
                throw std::logic_error{"Incorrect column indices"};
        };

        for (size_t local_row = 0; local_row < n_my_rows; ++local_row)
        {
            const auto global_row_ind = sparsity_graph->getRowMap()->getGlobalElement(local_row);
            sparsity_graph->getGlobalRowCopy(global_row_ind, view, row_size);
            check_row_global_cols(dense_graph.getRow(global_row_ind));
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        comm.abort();
    }
}
