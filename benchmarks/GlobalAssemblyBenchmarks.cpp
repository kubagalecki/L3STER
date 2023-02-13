#include "Common.hpp"
#include "DataPath.h"

static void BM_LocalDofIntervalComputation(benchmark::State& state)
{
    GlobalResource< KokkosScopeGuard >::getMaybeUninitialized();

    auto  read_mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto& part      = read_mesh.getPartitions()[0];
    part.initDualGraph();
    const auto mesh = convertMeshToOrder< 2 >(part);

    constexpr auto problemdef_ctwrapper = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{true, false}},
                                                                     Pair{d_id_t{2}, std::array{false, true}}} >{};

    for (auto _ : state)
        benchmark::DoNotOptimize(detail::computeLocalDofIntervals(mesh, problemdef_ctwrapper));
}
BENCHMARK(BM_LocalDofIntervalComputation)
    ->Name("Compute local DOF intervals")
    ->UseRealTime()
    ->Unit(benchmark::kMillisecond);

static void BM_GlobalDofMapComputation(benchmark::State& state)
{
    GlobalResource< KokkosScopeGuard >::getMaybeUninitialized();

    auto  read_mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto& part      = read_mesh.getPartitions()[0];
    part.initDualGraph();
    const auto mesh = convertMeshToOrder< 2 >(part);

    constexpr auto problemdef_ctwrapper = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{true, false}},
                                                                     Pair{d_id_t{2}, std::array{false, true}}} >{};

    const auto dof_intervals = detail::computeLocalDofIntervals(mesh, problemdef_ctwrapper);
    for (auto _ : state)
        benchmark::DoNotOptimize(NodeToGlobalDofMap{mesh, dof_intervals});
}
BENCHMARK(BM_GlobalDofMapComputation)->Name("Compute global DOF map")->UseRealTime()->Unit(benchmark::kMillisecond);

static void BM_SparsityPatternAssembly(benchmark::State& state)
{
    GlobalResource< KokkosScopeGuard >::getMaybeUninitialized();

    auto  read_mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto& part      = read_mesh.getPartitions()[0];
    part.initDualGraph();
    const auto mesh = convertMeshToOrder< 2 >(part);

    constexpr auto problemdef_ctwrapper = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{true, false}},
                                                                     Pair{d_id_t{2}, std::array{false, true}}} >{};

    const auto dof_intervals          = detail::computeLocalDofIntervals(mesh, problemdef_ctwrapper);
    const auto owned_plus_shared_dofs = detail::getNodeDofs(mesh.getAllNodes(), dof_intervals);
    const auto global_dof_map         = NodeToGlobalDofMap{mesh, dof_intervals};

    for (auto _ : state)
        benchmark::DoNotOptimize(
            detail::computeDofGraph(mesh, global_dof_map, owned_plus_shared_dofs, problemdef_ctwrapper));
}
BENCHMARK(BM_SparsityPatternAssembly)->Name("Sparsity pattern assembly")->UseRealTime()->Unit(benchmark::kMillisecond);

static void BM_OwnerOrSharedNodeDeterminationNotGhost(benchmark::State& state)
{
    auto       mesh            = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto&      part            = mesh.getPartitions()[0];
    const auto n_nodes_visited = part.transformReduce(
        0ul,
        std::plus<>{},
        [](const auto& element) { return element.getNodes().size(); },
        std::views::single(1),
        std::execution::par);

    part.initDualGraph();
    const auto n_parts = state.range(0);
    mesh               = partitionMesh(mesh, n_parts, {2});

    for (auto _ : state)
    {
        for (const auto& prt : mesh.getPartitions())
            prt.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                        benchmark::DoNotOptimize(not prt.isGhostNode(node));
                },
                std::views::single(1));
    }
    const auto nodes_processed = static_cast< double >(n_nodes_visited * state.iterations());
    state.counters["Query rate"] =
        benchmark::Counter{nodes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000};
}
BENCHMARK(BM_OwnerOrSharedNodeDeterminationNotGhost)
    ->Name("Is node owned or shared? [not ghost]")
    ->RangeMultiplier(2)
    ->Range(1, 1 << 4)
    ->Unit(benchmark::kMillisecond);

static void BM_OwnerOrSharedNodeDeterminationShared(benchmark::State& state)
{
    auto       mesh            = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto&      part            = mesh.getPartitions()[0];
    const auto n_nodes_visited = part.transformReduce(
        0ul,
        std::plus<>{},
        [](const auto& element) { return element.getNodes().size(); },
        std::views::single(1),
        std::execution::par);

    part.initDualGraph();
    const auto n_parts = state.range(0);
    mesh               = partitionMesh(mesh, n_parts, {2});

    for (auto _ : state)
    {
        for (const auto& prt : mesh.getPartitions())
            prt.visit(
                [&](const auto& element) {
                    for (auto node : element.getNodes())
                        benchmark::DoNotOptimize(prt.isOwnedNode(node));
                },
                std::views::single(1));
    }
    const auto nodes_processed = static_cast< double >(n_nodes_visited * state.iterations());
    state.counters["Query rate"] =
        benchmark::Counter{nodes_processed, benchmark::Counter::kIsRate, benchmark::Counter::kIs1000};
}
BENCHMARK(BM_OwnerOrSharedNodeDeterminationShared)
    ->Name("Is node owned or shared? [is owned]")
    ->RangeMultiplier(2)
    ->Range(1, 1 << 4)
    ->Unit(benchmark::kMillisecond);
