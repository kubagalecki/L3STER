#include "Common.hpp"
#include "DataPath.h"

void BM_MeshRead(benchmark::State& state)
{
    for (auto _ : state)
        const auto temp = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
}
BENCHMARK(BM_MeshRead)->Unit(benchmark::kMillisecond)->Name("Read mesh");

void BM_DualGraphGeneration(benchmark::State& state)
{
    auto  mesh      = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
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
    auto  mesh      = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
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
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);

    for (auto _ : state)
    {
        const auto boundary_view = mesh.getPartitions()[0].getBoundaryView(2);
        benchmark::DoNotOptimize(boundary_view);
    }
}
BENCHMARK(BM_BoundaryViewGenerationFallback)->Unit(benchmark::kSecond)->Name("Generate boundary view (fallback)");

void BM_MeshOrderConversion(benchmark::State& state)
{
    auto  mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto& part = mesh.getPartitions()[0];
    part.initDualGraph();

    for (auto _ : state)
    {
        const auto converted = convertMeshToOrder< 2 >(part);
        benchmark::DoNotOptimize(converted);
    }
}
BENCHMARK(BM_MeshOrderConversion)->Unit(benchmark::kSecond)->Name("Convert mesh to 2nd order");

void BM_MeshPartitioning(benchmark::State& state)
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);

    for (auto _ : state)
    {
        const auto parted = partitionMesh(mesh, 2, {});
        benchmark::DoNotOptimize(parted);
    }
}
BENCHMARK(BM_MeshPartitioning)->Unit(benchmark::kMillisecond)->Name("Partition mesh in half");

void BM_MeshSerialization(benchmark::State& state)
{
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    for (auto _ : state)
    {
        const auto serialized = SerializedPartition{mesh.getPartitions()[0]};
        benchmark::DoNotOptimize(serialized);
    }
}
BENCHMARK(BM_MeshSerialization)->Unit(benchmark::kMillisecond)->Name("Serialize mesh");

void BM_MeshDeserialization(benchmark::State& state)
{
    const auto mesh       = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    const auto serialized = SerializedPartition{mesh.getPartitions()[0]};
    for (auto _ : state)
    {
        const auto part2 = deserializePartition(serialized);
        benchmark::DoNotOptimize(part2);
    }
}
BENCHMARK(BM_MeshDeserialization)->Unit(benchmark::kMillisecond)->Name("Deserialize mesh");

template < typename ExecutionPolicy >
void BM_CopyElementNodes(benchmark::State& state)
{
    const auto            mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    const auto&           part = mesh.getPartitions()[0];
    std::atomic< size_t > progress_counter{0u};

    const auto element_op_counter = [&]< ElementTypes T, el_o_t O >(const Element< T, O >&) {
        progress_counter.fetch_add(Element< T, O >::n_nodes, std::memory_order_relaxed);
    };
    part.cvisit(element_op_counter, std::execution::par);

    const auto read_element = [&]< ElementTypes T, el_o_t O >(const Element< T, O >& el) {
        auto nodes_copy = el.getNodes();
        benchmark::DoNotOptimize(nodes_copy);
    };

    for (auto _ : state)
        part.cvisit(read_element, ExecutionPolicy{});

    state.SetBytesProcessed(progress_counter.load() * sizeof(n_id_t) * state.iterations());
}
BENCHMARK_TEMPLATE(BM_CopyElementNodes, std::execution::sequenced_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Make local node list copy [serial]");
BENCHMARK_TEMPLATE(BM_CopyElementNodes, std::execution::parallel_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Make local node list copy [parallel]");