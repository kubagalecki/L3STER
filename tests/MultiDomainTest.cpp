#include "l3ster/algsys/MakeAlgebraicSystem.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/solve/Amesos2Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"
#include "TestDataPath.h"

using namespace lstr;
using namespace lstr::algsys;

template < CondensationPolicy CP >
void test()
{
    constexpr auto domains     = std::array< d_id_t, 4 >{13, 14, 15, 16};
    auto           problem_def = ProblemDefinition< domains.size() >{};
    for (auto&& [i, dom] : domains | std::views::enumerate)
        problem_def.define({dom}, {i});

    const auto comm = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto mesh = readAndDistributeMesh< 2 >(
        *comm, L3STER_TESTDATA_ABSPATH(gmsh_ascii4_square_multidom.msh), mesh::gmsh_tag, {}, {}, problem_def);

    constexpr auto alg_params       = AlgebraicSystemParams{.cond_policy = CP, .n_rhs = 2};
    constexpr auto algparams_ctwrpr = util::ConstexprValue< alg_params >{};
    auto           alg_sys          = makeAlgebraicSystem(comm, mesh, problem_def, {}, algparams_ctwrpr);

    constexpr auto   ker_params = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1, .n_rhs = 2};
    constexpr double inc        = 1000.;
    double           set_value  = 1.;
    const auto       const_kernel =
        wrapDomainEquationKernel< ker_params >([&set_value]([[maybe_unused]] const auto& in, auto& out) {
            auto& [operators, rhs] = out;
            auto& [A0, A1, A2]     = operators;
            A0(0, 0)               = 1.;
            rhs(0, 0)              = set_value;
            rhs(0, 1)              = set_value + inc;
        });

    const auto assemble_problem_in_dom = [&]< int dom_ind >(util::ConstexprValue< dom_ind >) {
        set_value                 = static_cast< double >(dom_ind + 1);
        constexpr auto field_inds = std::array{size_t{dom_ind}};
        constexpr auto dom_ids    = std::array{domains[dom_ind]};
        alg_sys.assembleProblem(const_kernel, dom_ids, {}, util::ConstexprValue< field_inds >{});
    };

    alg_sys.beginAssembly();
    assemble_problem_in_dom(util::ConstexprValue< 0 >{});
    assemble_problem_in_dom(util::ConstexprValue< 1 >{});
    assemble_problem_in_dom(util::ConstexprValue< 2 >{});
    assemble_problem_in_dom(util::ConstexprValue< 3 >{});
    alg_sys.endAssembly();

    auto solver = solvers::Lapack{};
    alg_sys.solve(solver);

    auto solution_manager = SolutionManager{*mesh, domains.size() * 2};
    for (size_t i = 0; i != domains.size(); ++i)
    {
        solution_manager.setFields({2 * i}, static_cast< double >(i + 1));
        solution_manager.setFields({2 * i + 1}, static_cast< double >(i + 1) + inc);
    }
    constexpr auto dof_inds     = util::makeIotaArray< size_t, domains.size() >();
    constexpr auto sol_man_inds = util::makeIotaArray< size_t, domains.size() * 2 >();
    alg_sys.updateSolution(dof_inds, solution_manager, sol_man_inds);

    constexpr double eps = 1e-8;
    for (size_t i = 0; i != domains.size(); ++i)
    {
        const auto field_vals1 = solution_manager.getFieldView(2 * i);
        const auto field_vals2 = solution_manager.getFieldView(2 * i + 1);
        const auto v1          = static_cast< double >(i + 1);
        const auto v2          = static_cast< double >(i + 1) + inc;
        const auto check       = [&comm](auto view, double v) {
            REQUIRE(std::ranges::all_of(view, [&](double v_test) { return std::fabs(v - v_test) < eps; }));
            const char modified_local = std::ranges::any_of(view, [&](double v_test) { return v != v_test; });
            char       modified_global;
            comm->allReduce(std::span{&modified_local, 1}, &modified_global, MPI_LOR);
            REQUIRE(modified_global); // Check at least one entry was correctly written into
        };
        check(field_vals1, v1);
        check(field_vals2, v2);
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
