#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/global_assembly/SparsityPattern.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"

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
    const auto     sparsity_graph = detail::makeSparsityPattern(my_partition, problem_def, dof_intervals, comm);
    const auto     node_dof_map   = GlobalNodeToDofMap{my_partition, dof_intervals};
    const auto     row_map        = LocalNodeToDofMap{my_partition, dof_intervals};
    const auto     col_map        = LocalNodeToColumnDofMap{my_partition, sparsity_graph, node_dof_map};
    const auto     my_dofs        = detail::getNodeDofs(my_partition.getNodes(), dof_intervals);

    const auto                                n_my_rows     = sparsity_graph->getNodeNumRows();
    const auto                                n_global_cols = sparsity_graph->getGlobalNumCols();
    std::vector< std::vector< local_dof_t > > local_graph(n_my_rows, std::vector< local_dof_t >(n_global_cols));
    for (size_t local_row = 0; local_row < n_my_rows; ++local_row)
    {
        size_t row_size{};
        sparsity_graph->getLocalRowCopy(local_row, local_graph[local_row], row_size);
        local_graph[local_row].resize(row_size);
        std::ranges::sort(local_graph[local_row]);
    }

    const auto converter = NodeLocalGlobalConverter{my_partition};
    converter.convertToLocal(my_partition);

    const auto check_domain = [&]< auto dom_def >(ConstexprValue< dom_def >)
    {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto check_el_dofs = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& el) {
            const auto row_dofs = detail::getElementDofs< covered_dof_inds >(el, row_map);
            const auto col_dofs = detail::getElementDofs< covered_dof_inds >(el, col_map);

            for (auto row : row_dofs | std::views::filter(
                                           [&my_dofs](auto dof) { return std::ranges::binary_search(my_dofs, dof); }))
                for (auto col : col_dofs)
                    if (not std::ranges::binary_search(local_graph[row], col))
                        throw std::logic_error{"Expected entry not found"};
        };
        my_partition.cvisit(check_el_dofs, {domain_id});
    };
    try
    {
        forConstexpr(check_domain, problem_def);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        comm.abort();
    }
}
