#include "l3ster.hpp"

#include <benchmark/benchmark.h>

void BM_JacobianComputation(benchmark::State& state)
{
    using namespace lstr;
    Element< ElementTypes::Hex, 1 > el{{0, 1, 2, 3, 4, 5, 6, 7},
                                       ElementData< ElementTypes::Hex, 1 >{{Point{0., 0., 0.},
                                                                            Point{1., 0., 0.},
                                                                            Point{0., 1., 0.},
                                                                            Point{1., 1., 0.},
                                                                            Point{0., 0., 1.},
                                                                            Point{1., 0., 1.},
                                                                            Point{0., 1., 1.},
                                                                            Point{2., 2., 2.}}},
                                       0};
    for (auto _ : state)
    {
        const auto jacobi_mat_eval = getNatJacobiMatGenerator(el);
        const auto val             = jacobi_mat_eval(Point{.5, .5, .5});
        benchmark::DoNotOptimize(val);
    }
}

BENCHMARK(BM_JacobianComputation);
