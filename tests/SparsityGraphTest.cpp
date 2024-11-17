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
    template < el_o_t... orders, ProblemDef problem_def, CondensationPolicy CP >
    DenseGraph(const MpiComm&                          comm,
               const mesh::MeshPartition< orders... >& mesh,
               util::ConstexprValue< problem_def >     probdef_ctwrpr,
               const NodeCondensationMap< CP >&        cond_map)
    {
        using namespace std::views;
        const auto node_to_dof_map = NodeToGlobalDofMap{comm, mesh, cond_map, probdef_ctwrpr};
        const auto max_dof         = std::ranges::max(cond_map.getCondensedIds() | transform(node_to_dof_map) | join |
                                              filter(NodeToGlobalDofMap< problem_def.n_fields >::isValid));
        m_dim                      = static_cast< size_t >(max_dof + 1);
        m_entries                  = util::DynamicBitset{m_dim * m_dim};

        constexpr auto n_fields     = problem_def.n_fields;
        const auto     visit_domain = [&]< DomainDef< n_fields > dom_def >(util::ConstexprValue< dom_def >) {
            const auto visit_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& element) {
                const auto element_dofs = std::invoke([&] {
                    if constexpr (CP == CondensationPolicy::None)
                    {
                        constexpr auto covered_dof_inds = util::getTrueInds< dom_def.active_fields >();
                        return getUnsortedPrimaryDofs< covered_dof_inds >(element, node_to_dof_map, cond_map);
                    }
                    else if constexpr ((CP == CondensationPolicy::ElementBoundary))
                        return getUnsortedPrimaryDofs(element, node_to_dof_map, cond_map);
                });
                for (auto row : element_dofs)
                    for (auto col : element_dofs)
                        getRow(static_cast< size_t >(row)).set(static_cast< size_t >(col));
            };
            mesh.visit(visit_element, dom_def.domain);
        };
        util::forConstexpr(visit_domain, probdef_ctwrpr);
    }

    [[nodiscard]] auto getRow(size_t row) { return m_entries.getSubView(row * m_dim, (row + 1) * m_dim); }
    [[nodiscard]] auto getRow(size_t row) const { return m_entries.getSubView(row * m_dim, (row + 1) * m_dim); }

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
auto allGatherMesh(const MpiComm&                          comm,
                   const mesh::MeshPartition< orders... >& mesh) -> mesh::MeshPartition< orders... >
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
        constexpr int nodes_per_edge = 6;
        const auto    node_dist      = util::linspaceArray< nodes_per_edge >(0., 1.);
        return mesh::makeCubeMesh(node_dist);
    };
    constexpr el_o_t order        = 2;
    const auto       my_partition = generateAndDistributeMesh< order >(comm, mesh_generator);
    const auto       full_mesh    = allGatherMesh(comm, *my_partition);

    constexpr auto problem_def =
        ProblemDef{defineDomain< 2 >(0, 1), defineDomain< 2 >(1, 0), defineDomain< 2 >(3, 0, 1)};
    constexpr auto probdef_ctwrpr = util::ConstexprValue< problem_def >{};

    const auto cond_map_local = makeCondensationMap< CP >(comm, *my_partition, probdef_ctwrpr);
    const auto cond_map_full  = makeCondensationMap< CP >(comm_self, full_mesh, probdef_ctwrpr);
    const auto node_dof_map   = NodeToGlobalDofMap{comm, *my_partition, cond_map_local, probdef_ctwrpr};
    const auto sparsity_graph = makeSparsityGraph(comm, *my_partition, node_dof_map, cond_map_local, probdef_ctwrpr);

    const auto num_dofs_local    = node_dof_map.getNumOwnedDofs();
    auto       num_dofs_local_sv = std::views::single(num_dofs_local);
    size_t     num_dofs_global{};
    comm.allReduce(num_dofs_local_sv, &num_dofs_global, MPI_SUM);

    const auto dense_graph = DenseGraph{comm_self, full_mesh, probdef_ctwrpr, cond_map_full};

    REQUIRE(sparsity_graph->getLocalNumRows() == num_dofs_local);
    REQUIRE(sparsity_graph->getGlobalNumRows() == num_dofs_global);
    REQUIRE(sparsity_graph->getGlobalNumCols() == num_dofs_global);

    using inds_view_t                = tpetra_crsgraph_t::nonconst_global_inds_host_view_type;
    auto       view                  = inds_view_t("Global cols", sparsity_graph->getGlobalNumCols());
    size_t     row_size              = 0;
    const auto check_row_global_cols = [&](const auto& dense_row) {
        REQUIRE(row_size == dense_row.count());
        REQUIRE(std::ranges::all_of(util::asSpan(view).subspan(0, row_size),
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
