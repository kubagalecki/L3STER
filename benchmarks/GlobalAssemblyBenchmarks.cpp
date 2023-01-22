#include "Common.hpp"
#include "DataPath.h"

static void BM_SparsityPatternAssembly(benchmark::State& state)
{
    auto  read_mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto& part      = read_mesh.getPartitions()[0];
    part.initDualGraph();
    const auto mesh = convertMeshToOrder< 2 >(part);

    constexpr auto problem_def = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{true, false}},
                                                            Pair{d_id_t{2}, std::array{false, true}}} >{};

    const auto dof_intervals          = detail::computeLocalDofIntervals(mesh, problem_def);
    const auto owned_plus_shared_dofs = detail::getNodeDofs(mesh.getOwnedNodes(), dof_intervals);

    for (auto _ : state)
    {
        const auto entries = detail::calculateCrsData(mesh, problem_def, dof_intervals, owned_plus_shared_dofs);
        benchmark::DoNotOptimize(entries);
    }
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
