#include "DataPath.h"
#include "l3ster.hpp"

#include <benchmark/benchmark.h>

void BM_MeshRead(benchmark::State& state)
{
    for (auto _ : state)
        const auto temp = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);
}
BENCHMARK(BM_MeshRead)->Unit(benchmark::kMillisecond)->Name("Read mesh");

void BM_DualGraphGeneration(benchmark::State& state)
{
    auto  mesh      = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);
    auto& partition = mesh.getPartitions()[0];

    for (auto _ : state)
    {
        partition.initDualGraph();
        partition.deleteDualGraph();
    }
}
BENCHMARK(BM_DualGraphGeneration)->Unit(benchmark::kMillisecond)->Name("Generate dual graph");

void BM_BoundaryViewGeneration(benchmark::State& state)
{
    auto  mesh      = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);
    auto& partition = mesh.getPartitions()[0];
    partition.initDualGraph();

    for (auto _ : state)
    {
        const auto boundary_view = partition.getBoundaryView(2);
        benchmark::DoNotOptimize(boundary_view);
    }
}
BENCHMARK(BM_BoundaryViewGeneration)->Unit(benchmark::kMillisecond)->Name("Generate boundary view");

void BM_BoundaryViewGenerationFallback(benchmark::State& state)
{
    const auto mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);

    for (auto _ : state)
    {
        const auto boundary_view = mesh.getPartitions()[0].getBoundaryView(2);
        benchmark::DoNotOptimize(boundary_view);
    }
}
BENCHMARK(BM_BoundaryViewGenerationFallback)->Unit(benchmark::kSecond)->Name("Generate boundary view (fallback)");

void BM_MeshOrderConversion(benchmark::State& state)
{
    auto  mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);
    auto& part = mesh.getPartitions()[0];
    part.initDualGraph();

    for (auto _ : state)
    {
        const auto converted = lstr::convertMeshToOrder< 2 >(part);
        benchmark::DoNotOptimize(converted);
    }
}
BENCHMARK(BM_MeshOrderConversion)->Unit(benchmark::kSecond)->Name("Convert mesh to 2nd order");

void BM_MeshPartitioning(benchmark::State& state)
{
    auto mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);

    for (auto _ : state)
    {
        const auto parted = lstr::partitionMesh(mesh, 2, {});
        benchmark::DoNotOptimize(parted);
    }
}
BENCHMARK(BM_MeshPartitioning)->Unit(benchmark::kMillisecond)->Name("Partition mesh in half");

template < typename ExecutionPolicy >
void BM_ElementIteration(benchmark::State& state)
{
    const auto     mesh         = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);
    const auto&    part         = mesh.getPartitions()[0];
    constexpr auto read_element = []< lstr::ElementTypes T, lstr::el_o_t O >(const lstr::Element< T, O >& el) {
        const auto nodes_copy = el.getNodes();
        benchmark::DoNotOptimize(nodes_copy);
    };

    for (auto _ : state)
        part.cvisit(read_element, ExecutionPolicy{});
}
BENCHMARK_TEMPLATE(BM_ElementIteration, std::execution::sequenced_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Serial element iteration");
BENCHMARK_TEMPLATE(BM_ElementIteration, std::execution::parallel_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Parallel element iteration");
