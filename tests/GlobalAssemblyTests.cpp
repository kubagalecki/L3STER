#include "l3ster/global_assembly/SparsityPattern.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

using namespace lstr;

TEST_CASE("Sparsity pattern buffer", "[global_asm]")
{
    constexpr auto     n_rows = 1ull << 20, row_size = 1ull << 20;
    detail::CrsEntries crs_entries{n_rows};
    auto               test_buf = std::make_unique_for_overwrite< size_t[] >(row_size);

    std::mt19937                            prng{std::random_device{}()};
    std::uniform_int_distribution< size_t > distribution{0, n_rows - 1};

    const auto test_row  = distribution(prng);
    auto       write_buf = crs_entries.getRowEntriesForOverwrite(test_row, row_size);
    std::ranges::generate(write_buf, [&, i = 0]() mutable {
        const auto val = distribution(prng);
        test_buf[i++]  = val;
        return val;
    });

    const auto read_buf      = crs_entries.getRowEntries(test_row);
    const auto expected_vals = std::views::counted(test_buf.get(), row_size);
    CHECK(std::ranges::equal(read_buf, expected_vals));
}

TEST_CASE("Sparsity pattern assembly", "[global_asm]")
{
    constexpr std::array node_dist{0., 1., 2., 3.};
    auto                 mesh0 = makeCubeMesh(node_dist);
    auto&                part  = mesh0.getPartitions()[0];
    part.initDualGraph();
    auto mesh = convertMeshToOrder< 2 >(part);

    constexpr auto probdef_ctwrpr = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true, false}},
                                                               Pair{d_id_t{1}, std::array{false, true}},
                                                               Pair{d_id_t{2}, std::array{true, true}}} >{};

    const auto dof_intervals = detail::computeLocalDofIntervals(mesh, probdef_ctwrpr);
    const auto dofs          = detail::getNodeDofs(mesh.getNodes(), dof_intervals);
    const auto sparse_graph  = detail::calculateCrsData(mesh, probdef_ctwrpr, dof_intervals, dofs);

    const auto                   node_to_dof_map = GlobalNodeToDofMap{mesh, dof_intervals};
    std::vector< DynamicBitset > dense_graph(dofs.size(), dofs.size());
    const auto                   process_domain = [&]< auto dom_def >(ConstexprValue< dom_def >)
    {
        constexpr auto  domain_id        = dom_def.first;
        constexpr auto& coverage         = dom_def.second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto element_dofs = detail::getElementDofs< covered_dof_inds >(element, node_to_dof_map);
            for (auto row : element_dofs)
                for (auto col : element_dofs)
                    dense_graph[row][col] = true;
        };
        mesh.cvisit(process_element, {domain_id});
    };
    forConstexpr(process_domain, probdef_ctwrpr);

    for (size_t row = 0; row < sparse_graph.size(); ++row)
    {
        const auto row_dofs = sparse_graph.getRowEntries(row);
        for (auto col : row_dofs)
            CHECK(dense_graph[row].test(col));
        CHECK(dense_graph[row].count() == row_dofs.size());
    }
}
