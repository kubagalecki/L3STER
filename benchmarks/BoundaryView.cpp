#include "DataPath.h"
#include "l3ster.hpp"

#include <benchmark/benchmark.h>

void BM_BoundaryViewCreation(benchmark::State& state)
{
    const auto mesh = lstr::readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), lstr::gmsh_tag);

    for (auto _ : state)
        mesh.getPartitions()[0].getBoundaryView(2);
}
BENCHMARK(BM_BoundaryViewCreation)->Unit(benchmark::kMillisecond);
