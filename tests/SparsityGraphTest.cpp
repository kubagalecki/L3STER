#include "l3ster/algsys/SparsityGraph.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "TestDataPath.h"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::dofs;
using namespace lstr::algsys;

class DenseGraph
{
public:
    [[nodiscard]] auto getRow(size_t row) { return m_entries.getSubView(row * m_dim, (row + 1) * m_dim); }
    [[nodiscard]] auto getRow(size_t row) const { return m_entries.getSubView(row * m_dim, (row + 1) * m_dim); }

    template < CondensationPolicy CP, el_o_t... orders, size_t max_dofs_per_node >
    DenseGraph(const MpiComm&                                comm,
               const mesh::MeshPartition< orders... >&       mesh,
               const ProblemDefinition< max_dofs_per_node >& problem_def,
               CondensationPolicyTag< CP >                   cp = {})
    {
        using namespace std::views;
        const auto node_to_dof_map = NodeToGlobalDofMap{comm, mesh, problem_def, {}, cp};
        m_dim                      = node_to_dof_map.ownership().owned().size();
        m_entries                  = util::DynamicBitset{m_dim * m_dim};

        for (const auto& [domains, dof_bmp] : problem_def)
        {
            const auto dof_inds      = util::getTrueInds(dof_bmp);
            const auto dof_inds_span = std::span{dof_inds};
            const auto visit_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& element) {
                const auto element_dofs = getDofsCopy(node_to_dof_map, element, dof_inds_span, cp);
                for (auto row : element_dofs)
                    for (auto col : element_dofs)
                        getRow(static_cast< size_t >(row)).set(static_cast< size_t >(col));
            };
            mesh.visit(visit_element, domains);
        }
    }

private:
    size_t              m_dim; // assume square
    util::DynamicBitset m_entries;
};

template < el_o_t... orders >
auto combineParts(std::span< const mesh::MeshPartition< orders... > > parts) -> mesh::MeshPartition< orders... >
{
    auto combined_domains = typename mesh::MeshPartition< orders... >::domain_map_t{};
    for (const auto& part : parts)
        for (d_id_t domain_id : part.getDomainIds())
        {
            auto& domain = combined_domains[domain_id];
            part.visit([&](const auto& el) { mesh::pushToDomain(domain, el); }, domain_id, std::execution::seq);
        }
    auto boundaries = std::vector< d_id_t >{};
    std::ranges::copy(parts | std::views::transform([](const auto& part) { return part.getBoundaryIdsView(); }) |
                          std::views::join,
                      std::back_inserter(boundaries));
    util::sortRemoveDup(boundaries);
    return {std::move(combined_domains), boundaries};
}

template < el_o_t... orders >
auto allGatherMesh(const MpiComm& comm, const mesh::MeshPartition< orders... >& mesh)
    -> mesh::MeshPartition< orders... >
{
    using mesh_t = mesh::MeshPartition< orders... >;
    if (comm.getSize() == 1)
        return copy(mesh);
    auto reqs = std::vector< MpiComm::Request >{};
    if (comm.getRank() == 0)
    {
        auto parts    = util::ArrayOwner< mesh_t >(static_cast< size_t >(comm.getSize()));
        parts.front() = copy(mesh);
        for (int rank = 1; rank != comm.getSize(); ++rank)
            parts[static_cast< size_t >(rank)] = comm::receiveMesh< orders... >(comm, rank);
        const auto part_span = std::span{std::as_const(parts)};
        auto       combined  = combineParts(part_span);
        for (int rank = 1; rank != comm.getSize(); ++rank)
            comm::sendMesh(comm, combined, rank, std::back_inserter(reqs));
        MpiComm::Request::waitAll(reqs);
        return combined;
    }
    else
    {
        comm::sendMesh(comm, mesh, 0, std::back_inserter(reqs));
        MpiComm::Request::waitAll(reqs);
        return comm::receiveMesh< orders... >(comm, 0);
    }
}

template < CondensationPolicy CP >
void test(const MpiComm& comm)
{
    const auto     comm_self      = MpiComm{MPI_COMM_SELF};
    constexpr auto mesh_generator = [] {
        constexpr int nodes_per_edge = 5;
        const auto    node_dist      = util::linspaceArray< nodes_per_edge >(0., 1.);
        return mesh::makeCubeMesh(node_dist);
    };
    constexpr el_o_t order        = 2;
    const auto       my_partition = generateAndDistributeMesh< order >(comm, mesh_generator);
    const auto       full_mesh    = allGatherMesh(comm, *my_partition);

    auto problem_def = ProblemDefinition< 2 >{};
    problem_def.define({0}, {1});
    problem_def.define({1}, {0});
    problem_def.define({3}, {0, 1});

    constexpr auto cp_tag         = CondensationPolicyTag< CP >{};
    const auto     node_dof_map   = NodeToGlobalDofMap{comm, *my_partition, problem_def, {}, cp_tag};
    const auto     sparsity_graph = makeSparsityGraph(comm, *my_partition, node_dof_map, problem_def, cp_tag);

    const size_t num_dofs_local    = node_dof_map.ownership().owned().size();
    auto         num_dofs_local_sv = std::views::single(num_dofs_local);
    size_t       num_dofs_global{};
    comm.allReduce(num_dofs_local_sv, &num_dofs_global, MPI_SUM);

    const auto dense_graph = DenseGraph{comm_self, full_mesh, problem_def, cp_tag};

    REQUIRE(sparsity_graph->getLocalNumRows() == num_dofs_local);
    REQUIRE(sparsity_graph->getGlobalNumRows() == num_dofs_global);
    REQUIRE(sparsity_graph->getGlobalNumCols() == num_dofs_global);

    using inds_view_t                = tpetra_crsgraph_t::nonconst_global_inds_host_view_type;
    auto       view                  = inds_view_t("Global cols", sparsity_graph->getGlobalNumCols());
    size_t     row_size              = 0;
    const auto check_row_global_cols = [&](const auto& dense_row) {
        REQUIRE(row_size == dense_row.count());
        REQUIRE(std::ranges::all_of(util::asSpan(view).first(row_size),
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
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    const auto comm          = MpiComm{MPI_COMM_WORLD};

    test< CondensationPolicy::None >(comm);
    test< CondensationPolicy::ElementBoundary >(comm);
}
