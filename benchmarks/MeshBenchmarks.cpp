#include "Common.hpp"
#include "DataPath.h"

static void BM_MeshRead(benchmark::State& state)
{
    for (auto _ : state)
        benchmark::DoNotOptimize(readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {}, mesh::gmsh_tag));
}
BENCHMARK(BM_MeshRead)->Unit(benchmark::kMillisecond)->Name("Read mesh");

static void BM_DualGraphGeneration(benchmark::State& state)
{
    auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {}, mesh::gmsh_tag);
    for (auto _ : state)
        benchmark::DoNotOptimize(mesh::computeMeshDual(mesh, 2));
}
BENCHMARK(BM_DualGraphGeneration)->Unit(benchmark::kMillisecond)->Name("Generate dual graph");

static void BM_BoundaryViewGeneration(benchmark::State& state)
{
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {}, mesh::gmsh_tag);
    for (auto _ : state)
    {
        auto bnd_view = mesh::MeshPartition< 1 >::makeBoundaryElementViews(mesh, std::views::single(2));
        benchmark::DoNotOptimize(bnd_view);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_BoundaryViewGeneration)->Unit(benchmark::kMillisecond)->Name("Generate boundary view");

static void BM_MeshOrderConversion(benchmark::State& state)
{
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {}, mesh::gmsh_tag);
    for (auto _ : state)
    {
        auto converted = mesh::convertMeshToOrder< 2 >(mesh);
        benchmark::DoNotOptimize(converted);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_MeshOrderConversion)->Unit(benchmark::kSecond)->Name("Convert mesh to 2nd order");

static void BM_MeshPartitioning(benchmark::State& state)
{
    const auto mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {}, mesh::gmsh_tag);
    for (auto _ : state)
    {
        auto partd = partitionMesh(mesh, static_cast< idx_t >(state.range(0)));
        benchmark::DoNotOptimize(partd);
        benchmark::ClobberMemory();
    }
}
BENCHMARK(BM_MeshPartitioning)
    ->Unit(benchmark::kMillisecond)
    ->Name("Partition mesh")
    ->Unit(benchmark::kSecond)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 6);

template < typename ExecutionPolicy >
static void BM_CopyElementNodes(benchmark::State& state)
{
    const auto            mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {}, mesh::gmsh_tag);
    std::atomic< size_t > node_counter{0u};

    const auto element_op_counter = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >&) {
        node_counter.fetch_add(mesh::Element< T, O >::n_nodes, std::memory_order_relaxed);
    };
    mesh.visit(element_op_counter, std::execution::par);

    const auto read_element = [&]< mesh::ElementType T, el_o_t O >(const mesh::Element< T, O >& el) {
        const auto nodes = std::span{el.nodes};
        auto       hash  = robin_hood::hash_bytes(nodes.data(), nodes.size_bytes());
        benchmark::DoNotOptimize(hash);
    };

    constexpr auto policy = ExecutionPolicy{};
    for (auto _ : state)
        mesh.visit(read_element, policy);

    state.counters["Nodes hashed"] = benchmark::Counter{static_cast< double >(node_counter.load() * state.iterations()),
                                                        benchmark::Counter::kIsRate,
                                                        benchmark::Counter::kIs1000};
}
BENCHMARK_TEMPLATE(BM_CopyElementNodes, std::execution::sequenced_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Hash nodes [serial]");
BENCHMARK_TEMPLATE(BM_CopyElementNodes, std::execution::parallel_policy)
    ->Unit(benchmark::kMicrosecond)
    ->UseRealTime()
    ->Name("Hash nodes [parallel]");

static void BM_MakeLocalMeshView(benchmark::State& state)
{
    const d_id_t boundary_id = 2;
    const auto   mesh        = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {boundary_id}, mesh::gmsh_tag);

    for (auto _ : state)
    {
        auto local_view = mesh::LocalMeshView{mesh, mesh};
        benchmark::DoNotOptimize(local_view);
    }

    state.counters["Element processing rate"] =
        benchmark::Counter{static_cast< double >(mesh.getNElements() * state.iterations()),
                           benchmark::Counter::kIsRate,
                           benchmark::Counter::kIs1000};
}
BENCHMARK(BM_MakeLocalMeshView)->Unit(benchmark::kMillisecond)->Name("Generate local mesh view");

static void BM_SerializeMesh(benchmark::State& state)
{
    const d_id_t boundary_id = 2;
    const auto   mesh        = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {boundary_id}, mesh::gmsh_tag);
    size_t       serial_size{};

    for (auto _ : state)
    {
        auto serial = mesh::serializeMesh(mesh);
        serial_size = serial.size();
        benchmark::DoNotOptimize(serial.data());
        benchmark::ClobberMemory();
    }

    state.counters["Write B/s"] = benchmark::Counter{static_cast< double >(serial_size * state.iterations()),
                                                     benchmark::Counter::kIsRate,
                                                     benchmark::Counter::kIs1000};
}
BENCHMARK(BM_SerializeMesh)->Unit(benchmark::kMillisecond)->Name("Serialize mesh");

static void BM_DeserializeMesh(benchmark::State& state)
{
    const d_id_t boundary_id = 2;
    const auto   mesh        = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), {boundary_id}, mesh::gmsh_tag);
    const auto   serial      = mesh::serializeMesh(mesh);

    for (auto _ : state)
    {
        auto deser = mesh::deserializeMesh< 1 >(serial);
        benchmark::DoNotOptimize(deser);
    }

    state.counters["Read B/s"] = benchmark::Counter{static_cast< double >(serial.size() * state.iterations()),
                                                    benchmark::Counter::kIsRate,
                                                    benchmark::Counter::kIs1000};
}
BENCHMARK(BM_DeserializeMesh)->Unit(benchmark::kMillisecond)->Name("Deserialize mesh");
