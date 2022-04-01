#include "l3ster/global_assembly/SparsityPattern.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"

#include "TestDataPath.h"
#include "catch2/catch.hpp"

using namespace lstr;

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
    const auto process_domain = [&]< size_t dom_ind >(std::integral_constant< decltype(dom_ind), dom_ind >) {
        constexpr auto& problem_def = []< auto V >(ConstexprValue< V >)->const auto& { return V; }
        (probdef_ctwrpr);
        constexpr auto  domain_id        = problem_def[dom_ind].first;
        constexpr auto& coverage         = problem_def[dom_ind].second;
        constexpr auto  covered_dof_inds = getTrueInds< coverage >();

        const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
            const auto element_dofs = detail::getElementDofs< covered_dof_inds >(element, node_to_dof_map);
            for (auto row : element_dofs)
                for (auto col : element_dofs)
                    dense_graph[row][col] = true;
        };
        mesh.cvisit(process_element, {domain_id});
    };
    forConstexpr(process_domain,
                 std::make_index_sequence< []< auto V >(ConstexprValue< V >) { return V.size(); }(probdef_ctwrpr) >{});

    for (ptrdiff_t row = 0; const auto& row_dofs : sparse_graph)
    {
        for ([[maybe_unused]] auto col : row_dofs)
            CHECK(dense_graph[row].test(col));
        CHECK(dense_graph[row].count() == row_dofs.size());
        ++row;
    }
}
