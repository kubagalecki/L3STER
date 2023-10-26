#include "l3ster/glob_asm/SparsityGraph.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "TestDataPath.h"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::dofs;
using namespace lstr::glob_asm;

class DenseGraph
{
public:
    template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
    DenseGraph(const mesh::MeshPartition< orders... >&               mesh,
               util::ConstexprValue< problem_def >                   probdef_ctwrpr,
               const node_interval_vector_t< problem_def.n_fields >& dof_intervals,
               const NodeCondensationMap< CP >&                      cond_map)
    {
        const auto node_to_dof_map = dofs::NodeToGlobalDofMap{dof_intervals, cond_map};
        const auto max_dof =
            std::ranges::max(node_to_dof_map(cond_map.getCondensedIds().back()) | std::views::filter([](auto dof) {
                                 return dof != dofs::NodeToGlobalDofMap< problem_def.n_fields >::invalid_dof;
                             }));
        m_dim     = static_cast< size_t >(max_dof + 1);
        m_entries = util::DynamicBitset{m_dim * m_dim};

        constexpr auto n_fields = problem_def.n_fields;
        util::forConstexpr(
            [&]< DomainDef< n_fields > dom_def >(util::ConstexprValue< dom_def >) {
                const auto process_element = [&]< mesh::ElementType T, el_o_t O >(
                                                 const mesh::Element< T, O >& element) {
                    const auto element_dofs = std::invoke([&] {
                        if constexpr (CP == CondensationPolicy::None)
                        {
                            constexpr auto covered_dof_inds = util::getTrueInds< dom_def.active_fields >();
                            return dofs::getUnsortedPrimaryDofs< covered_dof_inds >(element, node_to_dof_map, cond_map);
                        }
                        else if constexpr ((CP == CondensationPolicy::ElementBoundary))
                            return dofs::getUnsortedPrimaryDofs(element, node_to_dof_map, cond_map);
                    });
                    for (auto row : element_dofs)
                        for (auto col : element_dofs)
                            getRow(static_cast< size_t >(row)).set(static_cast< size_t >(col));
                };
                mesh.visit(process_element, dom_def.domain);
            },
            probdef_ctwrpr);
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
void test(const MpiComm& comm)
{
    auto       full_mesh      = std::invoke([&comm] {
        constexpr int big = 16, small = 5;
        const auto    nodes_per_edge = comm.getSize() == 1 ? big : small;
        const auto    node_dist      = util::makeLinspaceVector(0., 1., nodes_per_edge);
        auto          mesh           = mesh::makeCubeMesh(node_dist);
        return convertMeshToOrder< 2 >(mesh);
    });
    auto       full_mesh_copy = copy(full_mesh);
    const auto my_partition   = distributeMesh(comm, std::move(full_mesh_copy));

    constexpr auto problem_def =
        ProblemDef{defineDomain< 2 >(0, 1), defineDomain< 2 >(1, 0), defineDomain< 2 >(3, 0, 1)};
    constexpr auto probdef_ctwrpr = util::ConstexprValue< problem_def >{};

    const auto cond_map       = makeCondensationMap< CP >(comm, *my_partition, probdef_ctwrpr);
    const auto cond_map_full  = makeCondensationMap< CP >(MpiComm{MPI_COMM_SELF}, full_mesh, probdef_ctwrpr);
    const auto dof_intervals  = computeDofIntervals(comm, *my_partition, cond_map, probdef_ctwrpr);
    const auto node_dof_map   = NodeToGlobalDofMap{dof_intervals, cond_map};
    const auto sparsity_graph = makeSparsityGraph(comm, *my_partition, node_dof_map, cond_map, probdef_ctwrpr);
    const auto num_all_dofs   = getNodeDofs(cond_map_full.getCondensedIds(), dof_intervals).size();
    const auto dense_graph    = DenseGraph{full_mesh, probdef_ctwrpr, dof_intervals, cond_map_full};

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
    // Set number of threads before initializing Kokkos
    const auto MpiGuard      = util::MpiScopeGuard{argc, argv};
    const auto comm          = MpiComm{MPI_COMM_WORLD};
    const auto n_threads     = 16 / comm.getSize();
    const auto max_par_guard = util::MaxParallelismGuard{static_cast< size_t >(n_threads)};

    const auto scope_guard = L3sterScopeGuard{argc, argv};

    // Test several times for the maximally parallel case to try to catch out any threading errors
    constexpr int n_mt_iters = 4;
    const int     n_iter     = comm.getSize() == 1 ? n_mt_iters : 1;
    for (int i = 0; i != n_iter; ++i)
    {
        test< CondensationPolicy::None >(comm);
        test< CondensationPolicy::ElementBoundary >(comm);
    }
}
