#include "Common.hpp"

template < q_o_t QO >
void BM_PhysBasisDersComputation(benchmark::State& state)
{
    constexpr auto QT = QuadratureTypes::GLeg;
    constexpr auto BT = BasisTypes::Lagrange;

    const auto element = getExampleHexElement();
    for (auto _ : state)
    {
        const auto ders = computePhysicalBasesAtQpoints< QT, QO, BT >(element);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 0)
    ->Name("Phys. basis der. at QPs computation QO0")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 2)
    ->Name("Phys. basis der. at QPs computation QO2")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 4)
    ->Name("Phys. basis der. at QPs computation QO4")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 6)
    ->Name("Phys. basis der. at QPs computation QO6")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 8)
    ->Name("Phys. basis der. at QPs computation QO8")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 10)
    ->Name("Phys. basis der. at QPs computation QO10")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 12)
    ->Name("Phys. basis der. at QPs computation QO12")
    ->Unit(benchmark::kMicrosecond);
BENCHMARK_TEMPLATE(BM_PhysBasisDersComputation, 14)
    ->Name("Phys. basis der. at QPs computation QO14")
    ->Unit(benchmark::kMicrosecond);
