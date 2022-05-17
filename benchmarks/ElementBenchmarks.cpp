#include "Common.hpp"

static void BM_JacobianComputation(benchmark::State& state)
{
    const auto element = getExampleHexElement< 1 >();
    const auto point   = Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(point);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_JacobianComputation)->Name("Compute Jacobian");

static void BM_ReferenceBasisComputation(benchmark::State& state)
{
    constexpr auto   T     = ElementTypes::Hex;
    constexpr el_o_t O     = 1;
    const auto       point = Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto ders = computeRefBasisDers< T, O, BasisTypes::Lagrange >(point);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_ReferenceBasisComputation)->Name("Compute reference basis");

static void BM_BasisPhysicalDerivativeComputation(benchmark::State& state)
{
    const auto element  = getExampleHexElement< 1 >();
    const auto point    = Point{.5, .5, .5};
    const auto J        = getNatJacobiMatGenerator(element)(point);
    const auto ref_ders = computeRefBasisDers< ElementTypes::Hex, 1, BasisTypes::Lagrange >(point);
    for (auto _ : state)
    {
        const auto ders = computePhysBasisDers(J, ref_ders);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_BasisPhysicalDerivativeComputation)->Name("Compute basis physical derivatives");
