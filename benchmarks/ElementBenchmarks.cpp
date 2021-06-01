#include "l3ster.hpp"

#include <benchmark/benchmark.h>

auto getExampleElement()
{
    using namespace lstr;
    Element< ElementTypes::Hex, 1 > element{{0, 1, 2, 3, 4, 5, 6, 7},
                                            ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                                 Point{1., 0., 0.},
                                                                                 Point{0., 1., 0.},
                                                                                 Point{1., 1., 0.},
                                                                                 Point{0., 0., 1.},
                                                                                 Point{1., 0., 1.},
                                                                                 Point{0., 1., 1.},
                                                                                 Point{2., 2., 2.}}},
                                            0};
    return element;
}

void BM_JacobianComputation(benchmark::State& state)
{
    const auto element = getExampleElement();
    const auto point   = lstr::Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(element);
        const auto val             = jacobi_mat_eval(point);
        benchmark::DoNotOptimize(val);
    }
}
BENCHMARK(BM_JacobianComputation);

void BM_ReferenceBasisComputation(benchmark::State& state)
{
    constexpr auto T     = lstr::ElementTypes::Hex;
    constexpr auto O     = lstr::el_o_t{1};
    const auto     point = lstr::Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto ders = lstr::computeRefBasisDers< T, O >(point);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_ReferenceBasisComputation);

void BM_SingleBasisDerivativeComputation(benchmark::State& state)
{
    const auto element = getExampleElement();
    const auto point   = lstr::Point{.5, .5, .5};
    for (auto _ : state)
    {
        const auto ders = computePhysBasisDers< 0 >(element, point);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_SingleBasisDerivativeComputation);

void BM_AggregateBasisDerivativeComputation(benchmark::State& state)
{
    const auto element  = getExampleElement();
    const auto point    = lstr::Point{.5, .5, .5};
    const auto J        = getNatJacobiMatGenerator(element)(point);
    const auto ref_ders = computeRefBasisDers< lstr::ElementTypes::Hex, 1 >(point);
    for (auto _ : state)
    {
        const auto ders = lstr::computePhysBasisDers< lstr::ElementTypes::Hex, 1 >(J, ref_ders);
        benchmark::DoNotOptimize(ders);
    }
}
BENCHMARK(BM_AggregateBasisDerivativeComputation);
