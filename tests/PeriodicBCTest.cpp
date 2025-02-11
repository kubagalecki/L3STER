#include "l3ster/bcs/PeriodicBC.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::mesh;
using namespace lstr::bcs;

constexpr auto node_dist = util::linspaceArray< 5 >(-1., 1.);
constexpr auto N         = node_dist.size();

template < size_t dim >
void assertCorrectNumEdges(const MpiComm& comm, const bcs::PeriodicBC< dim >& bc)
{
    auto dir_to_edge_local = std::array< size_t, dim >{};
    for (auto&& [from, to] : bc)
        for (size_t i = 0; i != dim; ++i)
            if (to[i] != invalid_node)
                ++dir_to_edge_local[i];
    auto dir_to_edge_global = std::array< size_t, dim >{};
    comm.allReduce(dir_to_edge_local, dir_to_edge_global.begin(), MPI_SUM);
    const auto boundary_size = dim == 2 ? N : N * N;
    REQUIRE(std::ranges::all_of(dir_to_edge_global, [&](size_t e) { return e == boundary_size; }));
}

void testBoundaryMatching2D()
{
    constexpr auto   dx        = N - 1;
    constexpr auto   dy        = dx * N;
    constexpr d_id_t domain_id = 0, bot_bound = 1, top_bound = 2, left_bound = 3, right_bound = 4;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 2 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    const auto       comm           = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto       mesh =
        generateAndDistributeMesh< 1 >(*comm, [&] { return makeSquareMesh(node_dist); }, {}, probdef_ctwrpr);
    auto period_def = PeriodicBCDefinition< 2 >{};
    period_def.definePeriodicBoundary({bot_bound}, {top_bound}, {0., node_dist.back() - node_dist.front(), 0.}, {0});
    period_def.definePeriodicBoundary({left_bound}, {right_bound}, {node_dist.back() - node_dist.front(), 0., 0.}, {1});
    const auto bc = PeriodicBC{period_def, *mesh, *comm};
    assertCorrectNumEdges(*comm, bc);
    if (comm->getSize() == 1) // Nodes get renumbered during partitioning, so we only know the result for comm size = 1
    {
        for (n_id_t n = 0; n != N; ++n)
            REQUIRE(bc.lookup(n)[0] == n + dy or bc.lookup(n + dy)[0] == n);
        for (n_id_t n = 0; n < N * N; n += N)
            REQUIRE(bc.lookup(n)[1] == n + dx or bc.lookup(n + dx)[1] == n);
    }
}

void testBoundaryMatching3D()
{
    constexpr auto   dx        = N - 1;
    constexpr auto   dy        = dx * N;
    constexpr auto   dz        = dy * N;
    constexpr d_id_t domain_id = 0;
    constexpr d_id_t bot_bound = 1, top_bound = 2, left_bound = 3, right_bound = 4, back_bound = 5, front_bound = 6;
    constexpr auto   problem_def    = ProblemDef{defineDomain< 3 >(domain_id, ALL_DOFS)};
    constexpr auto   probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    const auto       comm           = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto       mesh =
        generateAndDistributeMesh< 1 >(*comm, [&] { return makeCubeMesh(node_dist); }, {}, probdef_ctwrpr);
    auto period_def = PeriodicBCDefinition< 3 >{};
    period_def.definePeriodicBoundary({bot_bound}, {top_bound}, {0., 0., node_dist.back() - node_dist.front()}, {0});
    period_def.definePeriodicBoundary({left_bound}, {right_bound}, {0., node_dist.back() - node_dist.front(), 0.}, {1});
    period_def.definePeriodicBoundary({back_bound}, {front_bound}, {node_dist.back() - node_dist.front(), 0., 0.}, {2});
    const auto bc = PeriodicBC{period_def, *mesh, *comm};
    assertCorrectNumEdges(*comm, bc);
    if (comm->getSize() == 1) // Nodes get renumbered during partitioning, so we only know the result for comm size = 1
    {
        for (n_id_t x = 0; x != N; ++x)
            for (n_id_t y = 0; y != N; ++y)
            {
                const auto n1 = y * N + x;
                const auto n2 = n1 + dz;
                REQUIRE(bc.lookup(n1)[0] == n2 or bc.lookup(n2)[0] == n1);
            }
        for (n_id_t x = 0; x != N; ++x)
            for (n_id_t z = 0; z != N; ++z)
            {
                const auto n1 = z * N * N + x;
                const auto n2 = n1 + dy;
                REQUIRE(bc.lookup(n1)[1] == n2 or bc.lookup(n2)[1] == n1);
            }
        for (n_id_t y = 0; y != N; ++y)
            for (n_id_t z = 0; z != N; ++z)
            {
                const auto n1 = z * N * N + y * N;
                const auto n2 = n1 + dx;
                REQUIRE(bc.lookup(n1)[2] == n2 or bc.lookup(n2)[2] == n1);
            }
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{1};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    testBoundaryMatching2D();
    testBoundaryMatching3D();
}
