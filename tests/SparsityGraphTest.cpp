#include "l3ster/assembly/SparsityGraph.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "TestDataPath.h"

#include "Common.hpp"

using namespace lstr;

class DenseGraph
{
public:
    template < el_o_t... orders, auto problem_def, CondensationPolicy CP >
    DenseGraph(const mesh::MeshPartition< orders... >&                                     mesh,
               util::ConstexprValue< problem_def >                                         problemdef_ctwrpr,
               const detail::node_interval_vector_t< detail::deduceNFields(problem_def) >& dof_intervals,
               const detail::NodeCondensationMap< CP >&                                    cond_map)
    {
        const auto node_to_dof_map = NodeToGlobalDofMap{dof_intervals, cond_map};
        m_dim =
            std::ranges::max(node_to_dof_map(cond_map.getCondensedIds().back()) | std::views::filter([](auto dof) {
                                 return dof != NodeToGlobalDofMap< detail::deduceNFields(problem_def) >::invalid_dof;
                             })) +
            1;
        m_entries = util::DynamicBitset{m_dim * m_dim};

        const auto process_domain = [&]< auto dom_def >(util::ConstexprValue< dom_def >) {
            constexpr auto  domain_id        = dom_def.first;
            constexpr auto& coverage         = dom_def.second;
            constexpr auto  covered_dof_inds = util::getTrueInds< coverage >();

            const auto process_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& element) {
                const auto element_dofs = std::invoke([&] {
                    if constexpr (CP == CondensationPolicy::None)
                        return detail::getUnsortedPrimaryDofs< covered_dof_inds >(element, node_to_dof_map, cond_map);
                    else if constexpr ((CP == CondensationPolicy::ElementBoundary))
                        return detail::getUnsortedPrimaryDofs(element, node_to_dof_map, cond_map);
                });
                for (auto row : element_dofs)
                    for (auto col : element_dofs)
                        getRow(row).set(col);
            };
            mesh.visit(process_element, domain_id);
        };
        util::forConstexpr(process_domain, problemdef_ctwrpr);
    }

    [[nodiscard]] auto getRow(size_t row) { return m_entries.getSubView(row * m_dim, (row + 1) * m_dim); }
    [[nodiscard]] auto getRow(size_t row) const { return m_entries.getSubView(row * m_dim, (row + 1) * m_dim); }

    void print() const
    {
        for (size_t row = 0; row < m_dim; ++row)
        {
            const auto row_data = getRow(row);
            std::cout << row << ": ";
            for (size_t col = 0; col < m_dim; ++col)
                if (row_data.test(col))
                    std::cout << col << ' ';
            std::cout << '\n';
        }
    }

private:
    size_t              m_dim; // assume square
    util::DynamicBitset m_entries;
};

template < CondensationPolicy CP >
void test()
{
    const auto comm         = MpiComm{MPI_COMM_WORLD};
    const auto full_mesh    = std::invoke([] {
        constexpr auto node_dist = std::array{0., 1., 2., 3., 4.};
        auto           mesh      = mesh::makeCubeMesh(node_dist);
        mesh.initDualGraph();
        return convertMeshToOrder< 2 >(mesh);
    });
    const auto my_partition = distributeMesh(comm, full_mesh, {1, 3});

    constexpr auto problem_def    = std::array{util::Pair{d_id_t{0}, std::array{false, true}},
                                            util::Pair{d_id_t{1}, std::array{true, false}},
                                            util::Pair{d_id_t{3}, std::array{true, true}}};
    constexpr auto probdef_ctwrpr = util::ConstexprValue< problem_def >{};

    const auto cond_map       = detail::makeCondensationMap< CP >(comm, my_partition, probdef_ctwrpr);
    const auto cond_map_full  = detail::makeCondensationMap< CP >(MpiComm{MPI_COMM_SELF}, full_mesh, probdef_ctwrpr);
    const auto dof_intervals  = computeDofIntervals(comm, my_partition, cond_map, probdef_ctwrpr);
    const auto node_dof_map   = NodeToGlobalDofMap{dof_intervals, cond_map};
    const auto sparsity_graph = detail::makeSparsityGraph(comm, my_partition, node_dof_map, cond_map, probdef_ctwrpr);

    const auto num_all_dofs = detail::getNodeDofs(cond_map_full.getCondensedIds(), dof_intervals).size();
    const auto dense_graph  = DenseGraph{full_mesh, probdef_ctwrpr, dof_intervals, cond_map_full};

    REQUIRE(sparsity_graph->getGlobalNumRows() == num_all_dofs);
    REQUIRE(sparsity_graph->getGlobalNumCols() == num_all_dofs);

    auto view =
        tpetra_crsgraph_t::nonconst_global_inds_host_view_type("Global cols", sparsity_graph->getGlobalNumCols());
    size_t     row_size              = 0;
    const auto check_row_global_cols = [&](const auto& dense_row) {
        REQUIRE(row_size == dense_row.count());
        REQUIRE(std::ranges::all_of(std::views::counted(view.data(), row_size),
                                    [&](global_dof_t col_ind) { return dense_row.test(col_ind); }));
    };
    for (local_dof_t local_row = 0; static_cast< size_t >(local_row) != sparsity_graph->getLocalNumRows(); ++local_row)
    {
        const auto global_row_ind = sparsity_graph->getRowMap()->getGlobalElement(local_row);
        sparsity_graph->getGlobalRowCopy(global_row_ind, view, row_size);
        check_row_global_cols(dense_graph.getRow(global_row_ind));
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
