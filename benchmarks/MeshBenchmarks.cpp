#include "Common.hpp"
#include "DataPath.h"

static void BM_MeshRead(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag));
}
BENCHMARK(BM_MeshRead)->Unit(benchmark::kMillisecond)->Name("Read mesh");

static void BM_DualGraphGeneration(benchmark::State& state)
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    for (auto _ : state)
    {
        mesh.initDualGraph();
        state.PauseTiming();
        mesh.deleteDualGraph();
        state.ResumeTiming();
    }
}
BENCHMARK(BM_DualGraphGeneration)->Unit(benchmark::kMillisecond)->Name("Generate dual graph");

static void BM_BoundaryViewGeneration(benchmark::State& state)
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    mesh.initDualGraph();
    for (auto _ : state)
        benchmark::DoNotOptimize(mesh.getBoundaryView(2));
}
BENCHMARK(BM_BoundaryViewGeneration)->Unit(benchmark::kMillisecond)->Name("Generate boundary view");

static void BM_BoundaryViewGenerationFallback(benchmark::State& state)
{
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    for (auto _ : state)
        benchmark::DoNotOptimize(mesh.getBoundaryView(2));
}
BENCHMARK(BM_BoundaryViewGenerationFallback)->Unit(benchmark::kSecond)->Name("Generate boundary view (fallback)");

static void BM_MeshOrderConversion(benchmark::State& state)
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    mesh.initDualGraph();
    for (auto _ : state)
        benchmark::DoNotOptimize(mesh::convertMeshToOrder< 2 >(mesh));
}
BENCHMARK(BM_MeshOrderConversion)->Unit(benchmark::kSecond)->Name("Convert mesh to 2nd order");

static void BM_MeshPartitioning(benchmark::State& state)
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    mesh.initDualGraph();
    for (auto _ : state)
        benchmark::DoNotOptimize(partitionMesh(mesh, state.range(0), {2}));
}
BENCHMARK(BM_MeshPartitioning)
    ->Unit(benchmark::kMillisecond)
    ->Name("Partition mesh")
    ->Unit(benchmark::kSecond)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 6);

static void BM_MeshSerialization(benchmark::State& state)
{
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    for (auto _ : state)
        benchmark::DoNotOptimize(SerializedPartition{mesh});
}
BENCHMARK(BM_MeshSerialization)->Unit(benchmark::kMillisecond)->Name("Serialize mesh");

static void BM_MeshDeserialization(benchmark::State& state)
{
    const auto mesh       = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    const auto serialized = SerializedPartition{mesh};
    for (auto _ : state)
        benchmark::DoNotOptimize(deserializePartition< 1 >(serialized));
}
BENCHMARK(BM_MeshDeserialization)->Unit(benchmark::kMillisecond)->Name("Deserialize mesh");

template < typename ExecutionPolicy >
static void BM_CopyElementNodes(benchmark::State& state)
{
    const auto            mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), mesh::gmsh_tag);
    std::atomic< size_t > node_counter{0u};

    const auto element_op_counter = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >&) {
        node_counter.fetch_add(mesh::Element< T, O >::n_nodes, std::memory_order_relaxed);
    };
    mesh.visit(element_op_counter, std::execution::par);

    const auto read_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& el) {
        const auto nodes_copy = el.getNodes();
        benchmark::DoNotOptimize(nodes_copy);
    };

    for (auto _ : state)
        mesh.visit(read_element, ExecutionPolicy{});

    state.counters["Nodes read"] = benchmark::Counter{static_cast< double >(node_counter.load() * state.iterations()),
                                                      benchmark::Counter::kIsRate,
                                                      benchmark::Counter::kIs1000};
}
BENCHMARK_TEMPLATE(BM_CopyElementNodes, std::execution::sequenced_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Loop over all elements [serial]");
BENCHMARK_TEMPLATE(BM_CopyElementNodes, std::execution::parallel_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Loop over all elements [parallel]");