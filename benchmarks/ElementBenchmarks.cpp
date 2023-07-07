#include "Common.hpp"

static void BM_JacobianComputation(benchmark::State& state)
{
    const auto element = getExampleHexElement< 1 >();
    const auto point   = Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto jacobi_mat_eval = map::getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(point);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_JacobianComputation)->Name("Compute Jacobian [hex]");

static void BM_ReferenceBasisComputation(benchmark::State& state)
{
    constexpr auto   T     = ElementType::Hex;
    constexpr el_o_t O     = 1;
    const auto       point = Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto ders = basis::computeRefBasisDers< T, O, basis::BasisType::Lagrange >(point);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_ReferenceBasisComputation)->Name("Compute reference basis [hex 1]");

static void BM_BasisPhysicalDerivativeComputation(benchmark::State& state)
{
    const auto element  = getExampleHexElement< 1 >();
    const auto point    = Point{.5, .5, .5};
    const auto J        = map::getNatJacobiMatGenerator(element)(point);
    const auto ref_ders = basis::computeRefBasisDers< ElementType::Hex, 1, basis::BasisType::Lagrange >(point);
    for (auto _ : state)
    {
        const auto ders = map::computePhysBasisDers(J, ref_ders);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_BasisPhysicalDerivativeComputation)->Name("Compute basis physical derivatives [hex 1]");
