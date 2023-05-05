#include "Common.hpp"
#include "DataPath.h"

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
