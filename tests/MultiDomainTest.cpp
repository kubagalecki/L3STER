#include "l3ster/assembly/AlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"
#include "TestDataPath.h"

using namespace lstr;

template < CondensationPolicy CP >
void test()
{
    static constexpr auto domains        = std::array< d_id_t, 4 >{13, 14, 15, 16};
    constexpr auto        problem_def    = std::invoke([] {
        auto retval = std::array< Pair< d_id_t, std::array< bool, domains.size() > >, domains.size() >{};
        for (size_t i = 0; auto& [dom, cov] : retval)
        {
            dom      = domains[i];
            cov[i++] = true;
        }
        return retval;
    });
    constexpr auto        probdef_ctwrpr = ConstexprValue< problem_def >{};

    const auto comm = MpiComm{MPI_COMM_WORLD};
    const auto mesh = readAndDistributeMesh(comm,
                                            L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square_multidom.msh),
                                            gmsh_tag,
                                            {},
                                            ConstexprValue< el_o_t{2} >{},
                                            probdef_ctwrpr);

    auto alg_sys = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr);

    double     set_value = 1.;
    const auto const_kernel =
        [&set_value](const auto&, const std::array< std::array< val_t, 0 >, 2 >&, const SpaceTimePoint&) noexcept {
            auto retval = std::pair< std::array< Eigen::Matrix< val_t, 1, 1 >, 3 >, Eigen::Vector< val_t, 1 > >{};
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;
            A0(0, 0)              = 1.;
            rhs[0]                = set_value;
            A1.setZero();
            A2.setZero();
            return retval;
        };
    static_assert(detail::Kernel_c< decltype(const_kernel), 2, 0 >);

#define ASSEMBLE_CONST_PROBLEM(DOMAIN)                                                                                 \
    set_value = static_cast< double >(DOMAIN + 1);                                                                     \
    alg_sys->assembleDomainProblem(const_kernel,                                                                       \
                                   mesh,                                                                               \
                                   std::views::single(domains[DOMAIN]),                                                \
                                   empty_field_val_getter,                                                             \
                                   ConstexprValue< std::array{size_t{DOMAIN}} >{});

    std::puts("Made it to beginAssembly");
    alg_sys->beginAssembly();
    std::puts("Made it to assembly");
    ASSEMBLE_CONST_PROBLEM(0);
    ASSEMBLE_CONST_PROBLEM(1);
    ASSEMBLE_CONST_PROBLEM(2);
    ASSEMBLE_CONST_PROBLEM(3);
    std::puts("Made it to endAssembly");
    alg_sys->endAssembly(mesh);
    alg_sys->describe(comm);

    auto solver   = solvers::Lapack{};
    auto solution = alg_sys->makeSolutionVector();
    alg_sys->solve(solver, solution);

    auto solution_manager = SolutionManager{mesh, detail::deduceNFields(problem_def)};
    for (size_t i = 0; i != detail::deduceNFields(problem_def); ++i)
        solution_manager.setField(i, static_cast< double >(i + 1));
    constexpr auto dof_inds = makeIotaArray< size_t, detail::deduceNFields(problem_def) >();
    alg_sys->updateSolution(mesh, solution, dof_inds, solution_manager, dof_inds);

    for (size_t i = 0; i != detail::deduceNFields(problem_def); ++i)
    {
        const auto field_vals = solution_manager.getFieldView(i);
        REQUIRE(std::ranges::all_of(field_vals,
                                    [&](double v) { return std::fabs(v - static_cast< double >(i + 1) < 1e-8); }));
    }
}

int main(int argc, char* argv[])
{
    L3sterScopeGuard scope_guard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
