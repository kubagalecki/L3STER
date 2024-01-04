#include "Common.hpp"

static auto kernelImpl(size_t in, size_t rand) -> size_t
{
    thread_local const auto hash = std::hash< size_t >{};
    return hash(hash(hash(in) % rand)) * 953u + rand + 1u;
}

static void BM_ParallelNonRandomAccessForBaseline(benchmark::State& state)
{
    const auto size = static_cast< size_t >(state.range(0));

    auto prng = std::mt19937{std::random_device{}()};
    auto dist = std::uniform_int_distribution< size_t >{1, 1000};

    auto input  = robin_hood::unordered_flat_map< size_t, size_t >{};
    auto output = std::vector< size_t >(size);

    for (size_t i = 0; i != size; ++i)
        input.emplace(i, dist(prng));
    const auto kernel = [random_number = dist(prng)](size_t in) {
        return kernelImpl(in, random_number); // Arbitrary non-trivial operation
    };

    for (auto _ : state)
    {
        std::ranges::for_each(input, [&](const auto& pair) { output[pair.first] = kernel(pair.second); });
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.counters["Ops/s"] = benchmark::Counter{
        static_cast< double >(state.iterations() * size), benchmark::Counter::kIsRate, benchmark::Counter::kIs1000};
}
BENCHMARK(BM_ParallelNonRandomAccessForBaseline)
    ->Name("Parallel non-random access for - serial baseline")
    ->Unit(benchmark::kMillisecond)
    ->RangeMultiplier(16)
    ->Range(1 << 10, 1 << 22)
    ->UseRealTime();

static void BM_ParallelNonRandomAccessFor(benchmark::State& state)
{
    const auto size     = static_cast< size_t >(state.range(0));
    const auto maxpar   = static_cast< size_t >(state.range(1));
    const auto tbb_ctrl = oneapi::tbb::global_control{oneapi::tbb::global_control::max_allowed_parallelism, maxpar};

    auto prng = std::mt19937{std::random_device{}()};
    auto dist = std::uniform_int_distribution< size_t >{1, 1000};

    auto input  = robin_hood::unordered_flat_map< size_t, size_t >{};
    auto output = std::vector< size_t >(size);

    for (size_t i = 0; i != size; ++i)
        input.emplace(i, dist(prng));
    const auto kernel = [random_number = dist(prng)](size_t in) {
        return kernelImpl(in, random_number); // Arbitrary non-trivial operation
    };

    for (auto _ : state)
    {
        util::tbb::parallelFor(input, [&](const auto& pair) { output[pair.first] = kernel(pair.second); });
        benchmark::DoNotOptimize(output.data());
        benchmark::ClobberMemory();
    }

    state.counters["Ops/s"] = benchmark::Counter{
        static_cast< double >(state.iterations() * size), benchmark::Counter::kIsRate, benchmark::Counter::kIs1000};
}
BENCHMARK(BM_ParallelNonRandomAccessFor)
    ->Name("Parallel non-random access for")
    ->Unit(benchmark::kMillisecond)
    ->ArgsProduct({benchmark::CreateRange(1 << 10, 1 << 22, 16), benchmark::CreateRange(2, 16, 2)})
    ->UseRealTime();