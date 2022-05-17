#include "Common.hpp"
#include "DataPath.h"

static void BM_SparsityPatternAssembly(benchmark::State& state)
{
    auto  read_mesh = readMesh(L3STER_TESTDATA_ABSPATH(sphere.msh), gmsh_tag);
    auto& part      = read_mesh.getPartitions()[0];
    part.initDualGraph();
    const auto mesh = convertMeshToOrder< 2 >(part);

    constexpr auto problem_def = ConstexprValue< std::array{Pair{d_id_t{1}, std::array{true, false}},
                                                            Pair{d_id_t{2}, std::array{false, true}}} >{};

    const auto dof_intervals          = detail::computeLocalDofIntervals(mesh, problem_def);
    const auto owned_plus_shared_dofs = detail::getNodeDofs(mesh.getNodes(), dof_intervals);

    for (auto _ : state)
    {
        const auto entries = detail::calculateCrsData(mesh, problem_def, dof_intervals, owned_plus_shared_dofs);
        benchmark::DoNotOptimize(entries);
    }
}
BENCHMARK(BM_SparsityPatternAssembly)->Name("Sparsity pattern assembly")->UseRealTime()->Unit(benchmark::kMillisecond);
