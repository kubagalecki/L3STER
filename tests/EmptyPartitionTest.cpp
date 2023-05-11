#include "l3ster/assembly/AlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;

template < CondensationPolicy CP >
void test()
{
    constexpr d_id_t domain_id      = 0;
    constexpr auto   problem_def    = std::array{Pair{domain_id, std::array{true}}};
    constexpr auto   probdef_ctwrpr = ConstexprValue< problem_def >{};

    const auto comm = MpiComm{MPI_COMM_WORLD};
    const auto mesh = std::invoke([&] {
        if (comm.getRank() == 0)
        {
            constexpr el_o_t mesh_order = 2;
            auto             full_mesh  = makeSquareMesh(std::vector{0., 1.});
            full_mesh.getPartitions().front().initDualGraph();
            full_mesh.getPartitions().front() = convertMeshToOrder< mesh_order >(full_mesh.getPartitions().front());
            return distributeMesh(comm, full_mesh, {1, 2, 3, 4}, probdef_ctwrpr);
        }
        else
            return distributeMesh(comm, {}, {}, probdef_ctwrpr);
    });

    auto alg_sys = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr);

    const auto const_kernel =
        [](const auto&, const std::array< std::array< val_t, 0 >, 2 >&, const SpaceTimePoint&) noexcept {
            auto retval = std::pair< std::array< Eigen::Matrix< val_t, 1, 1 >, 3 >, Eigen::Vector< val_t, 1 > >{};
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;
            A0(0, 0)              = 1.;
            rhs[0]                = 1.;
            A1.setZero();
            A2.setZero();
            return retval;
        };
    static_assert(detail::Kernel_c< decltype(const_kernel), 2, 0 >);

    alg_sys->beginAssembly();
    alg_sys->assembleDomainProblem(const_kernel, mesh, std::views::single(domain_id));
    alg_sys->endAssembly(mesh);

    auto solver   = solvers::Lapack{};
    auto solution = alg_sys->makeSolutionVector();
    alg_sys->solve(solver, solution);

    constexpr auto n_fields         = detail::deduceNFields(problem_def);
    auto           solution_manager = SolutionManager{mesh, n_fields};
    constexpr auto dof_inds         = makeIotaArray< size_t, n_fields >();
    alg_sys->updateSolution(mesh, solution, dof_inds, solution_manager, dof_inds);
    REQUIRE(std::ranges::all_of(solution_manager.getFieldView(0), [&](double v) { return std::fabs(v - 1.) < 1e-10; }));
}

int main(int argc, char* argv[])
{
    L3sterScopeGuard scope_guard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
