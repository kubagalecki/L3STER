#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/glob_asm/AlgebraicSystem.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"
#include "TestDataPath.h"

using namespace lstr;
using namespace lstr::glob_asm;

template < CondensationPolicy CP >
void test()
{
    constexpr auto domains        = std::array< d_id_t, 4 >{13, 14, 15, 16};
    constexpr auto problem_def    = ProblemDef{defineDomain< domains.size() >(domains[0], 0),
                                            defineDomain< domains.size() >(domains[1], 1),
                                            defineDomain< domains.size() >(domains[2], 2),
                                            defineDomain< domains.size() >(domains[3], 3)};
    constexpr auto probdef_ctwrpr = util::ConstexprValue< problem_def >{};

    const auto comm = MpiComm{MPI_COMM_WORLD};
    const auto mesh = readAndDistributeMesh(comm,
                                            L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square_multidom.msh),
                                            mesh::gmsh_tag,
                                            {},
                                            util::ConstexprValue< el_o_t{2} >{},
                                            probdef_ctwrpr);

    auto alg_sys = makeAlgebraicSystem(comm, mesh, CondensationPolicyTag< CP >{}, probdef_ctwrpr);

    constexpr auto ker_params = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1};
    double         set_value  = 1.;
    const auto const_kernel = wrapDomainKernel< ker_params >([&set_value]([[maybe_unused]] const auto& in, auto& out) {
        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;
        A0(0, 0)               = 1.;
        rhs[0]                 = set_value;
    });

    const auto assemble_problem_in_dom = [&]< int dom_ind >(util::ConstexprValue< dom_ind >) {
        set_value                 = static_cast< double >(dom_ind + 1);
        constexpr auto field_inds = std::array{size_t{dom_ind}};
        constexpr auto dom_ids    = std::array{domains[dom_ind]};
        alg_sys->assembleDomainProblem(
            const_kernel, dom_ids, empty_field_val_getter, util::ConstexprValue< field_inds >{});
    };

    alg_sys->beginAssembly();
    assemble_problem_in_dom(util::ConstexprValue< 0 >{});
    assemble_problem_in_dom(util::ConstexprValue< 1 >{});
    assemble_problem_in_dom(util::ConstexprValue< 2 >{});
    assemble_problem_in_dom(util::ConstexprValue< 3 >{});
    alg_sys->endAssembly();

    auto solver   = solvers::Lapack{};
    auto solution = alg_sys->makeSolutionVector();
    alg_sys->solve(solver, solution);

    auto solution_manager = SolutionManager{*mesh, problem_def.n_fields};
    for (size_t i = 0; i != problem_def.n_fields; ++i)
        solution_manager.setField(i, static_cast< double >(i + 1));
    constexpr auto dof_inds = util::makeIotaArray< size_t, problem_def.n_fields >();
    alg_sys->updateSolution(solution, dof_inds, solution_manager, dof_inds);

    for (size_t i = 0; i != problem_def.n_fields; ++i)
    {
        const auto field_vals = solution_manager.getFieldView(i);
        REQUIRE(std::ranges::all_of(field_vals,
                                    [&](double v) { return std::fabs(v - static_cast< double >(i + 1)) < 1e-10; }));
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
