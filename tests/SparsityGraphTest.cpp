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
    template < auto problem_def >
    DenseGraph(const MeshPartition&                                                        mesh,
               ConstexprValue< problem_def >                                               problemdef_ctwrpr,
               const detail::node_interval_vector_t< detail::deduceNFields(problem_def) >& dof_intervals,
               const detail::NodeCondensationMap&                                          cond_map)
        : m_dim{cond_map.size()}, m_entries{m_dim * m_dim}
    {
        const auto node_to_dof_map = NodeToGlobalDofMap{mesh, dof_intervals, cond_map};
        const auto process_domain  = [&]< auto dom_def >(ConstexprValue< dom_def >) {
            constexpr auto domain_id        = dom_def.first;
            constexpr auto coverage         = dom_def.second;
            constexpr auto covered_dof_inds = getTrueInds< coverage >();

            const auto process_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                const auto element_dofs = detail::getSortedElementDofs< covered_dof_inds >(element, node_to_dof_map);
                for (auto row : element_dofs)
                    for (auto col : element_dofs)
                        getRow(row).set(col);
            };
            mesh.visit(process_element, domain_id);
        };
        forConstexpr(process_domain, problemdef_ctwrpr);
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
    size_t        m_dim; // assume square
    DynamicBitset m_entries;
};

int main(int argc, char* argv[])
{
    L3sterScopeGuard scope_guard{argc, argv};
    const MpiComm    comm{MPI_COMM_WORLD};

    Mesh          mesh;
    MeshPartition full_mesh;
    if (comm.getRank() == 0)
    {
        constexpr std::array node_dist{0., 1., 2., 3., 4.};
        mesh = makeCubeMesh(node_dist);
        mesh.getPartitions().front().initDualGraph();
        full_mesh                   = convertMeshToOrder< 2 >(mesh.getPartitions().front());
        const auto full_mesh_serial = SerializedPartition{full_mesh};
        for (int dest_rank = 1; dest_rank < comm.getSize(); ++dest_rank)
            sendPartition(comm, full_mesh_serial, dest_rank);
    }
    else
    {
        const auto full_mesh_serial = receivePartition(comm, 0);
        full_mesh                   = deserializePartition(full_mesh_serial);
    }
    const auto my_partition = distributeMesh(comm, mesh, {});

    constexpr auto problem_def    = std::array{Pair{d_id_t{0}, std::array{false, true}},
                                            Pair{d_id_t{1}, std::array{true, false}},
                                            Pair{d_id_t{3}, std::array{true, true}}};
    constexpr auto probdef_ctwrpr = ConstexprValue< problem_def >{};
    const auto     cond_map =
        detail::NodeCondensationMap::makeBoundaryNodeCondensationMap(comm, my_partition, probdef_ctwrpr);
    const auto dof_intervals  = computeDofIntervals(comm, my_partition, cond_map, probdef_ctwrpr);
    const auto global_dof_map = NodeToGlobalDofMap{my_partition, dof_intervals, cond_map};
    const auto sparsity_graph = detail::makeSparsityGraph(my_partition, global_dof_map, probdef_ctwrpr, comm);

    const auto all_dofs    = detail::getNodeDofs(full_mesh.getAllNodes(), dof_intervals);
    const auto dense_graph = DenseGraph{full_mesh, probdef_ctwrpr, dof_intervals, all_dofs.size()};

    REQUIRE(sparsity_graph->getGlobalNumRows() == all_dofs.size());
    REQUIRE(sparsity_graph->getGlobalNumCols() == all_dofs.size());

    const auto n_my_rows = sparsity_graph->getLocalNumRows();
    auto       view =
        tpetra_crsgraph_t::nonconst_global_inds_host_view_type("Global cols", sparsity_graph->getGlobalNumCols());
    size_t row_size = 0;

    const auto check_row_global_cols = [&](const auto& dense_row) {
        REQUIRE(row_size == dense_row.count());
        REQUIRE(std::ranges::all_of(std::views::counted(view.data(), row_size),
                                    [&](global_dof_t col_ind) { return dense_row.test(col_ind); }));
    };

    for (size_t local_row = 0; local_row < n_my_rows; ++local_row)
    {
        const auto global_row_ind = sparsity_graph->getRowMap()->getGlobalElement(local_row);
        sparsity_graph->getGlobalRowCopy(global_row_ind, view, row_size);
        check_row_global_cols(dense_graph.getRow(global_row_ind));
    }
}
