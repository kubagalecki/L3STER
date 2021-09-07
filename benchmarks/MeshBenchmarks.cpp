#include "DataPath.h"
#include "l3ster.hpp"

#include <benchmark/benchmark.h>

void BM_MeshRead(benchmark::State& state)
{
    for (auto _ : state)
        const auto temp = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);
}
BENCHMARK(BM_MeshRead)->Unit(benchmark::kMillisecond);

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
BENCHMARK(BM_DualGraphGeneration)->Unit(benchmark::kMillisecond);

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
BENCHMARK(BM_BoundaryViewGeneration)->Unit(benchmark::kMillisecond);

void BM_BoundaryViewGenerationFallback(benchmark::State& state)
{
    const auto mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);

    for (auto _ : state)
    {
        const auto boundary_view = mesh.getPartitions()[0].getBoundaryView(2);
        benchmark::DoNotOptimize(boundary_view);
    }
}
BENCHMARK(BM_BoundaryViewGenerationFallback)->Unit(benchmark::kSecond);

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
BENCHMARK(BM_MeshOrderConversion)->Unit(benchmark::kSecond);

void BM_MeshPartitioning(benchmark::State& state)
{
    auto mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);

    for (auto _ : state)
    {
        const auto parted = lstr::partitionMesh(mesh, 2, {});
        benchmark::DoNotOptimize(parted);
    }
}
BENCHMARK(BM_MeshPartitioning)->Unit(benchmark::kMillisecond);
