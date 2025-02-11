#include "l3ster/l3ster.hpp"

#include "DataPath.h"

using namespace lstr;
using namespace std::string_view_literals;

auto makeMesh(const lstr::MpiComm& comm, auto probdef_ctwrpr)
{
    constexpr auto node_dist  = std::invoke([] {
        constexpr size_t                   edge_divs = 6;
        constexpr auto                     dx        = 1. / static_cast< val_t >(edge_divs);
        std::array< val_t, edge_divs + 1 > retval{};
        for (double x = 0; auto& r : retval)
        {
            r = x;
            x += dx;
        }
        return retval;
    });
    constexpr auto mesh_order = 6;
    return generateAndDistributeMesh< mesh_order >(
        comm, [&] { return mesh::makeCubeMesh(node_dist); }, {}, probdef_ctwrpr);
}

template < CondensationPolicy CP, OperatorEvaluationStrategy S >
void solveDiffusion3DProblem()
{
    const auto comm = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    constexpr d_id_t      domain_id      = 0;
    static constexpr auto problem_def    = ProblemDef{defineDomain< 4 >(domain_id, ALL_DOFS)};
    constexpr auto        probdef_ctwrpr = util::ConstexprValue< problem_def >{};
    static constexpr auto boundary_ids   = util::makeIotaArray< d_id_t, 6 >(1);
    auto                  bc_def         = BCDefinition< problem_def.n_fields >{};
    bc_def.defineDirichlet(boundary_ids, {0});

    const auto my_partition = makeMesh(*comm, probdef_ctwrpr);

    constexpr auto field_inds  = util::makeIotaArray< size_t, problem_def.n_fields >();
    constexpr auto T_inds      = std::array< size_t, 1 >{0};
    constexpr auto T_grad_inds = std::array< size_t, 3 >{1, 2, 3};
    constexpr auto dof_inds    = field_inds;

    constexpr auto dom_params         = KernelParams{.dimension = 3, .n_equations = 7, .n_unknowns = 4};
    constexpr auto diffusion_kernel3d = wrapDomainEquationKernel< dom_params >([](const auto&, auto& out) {
        auto& [operators, rhs] = out;
        auto& [A0, Ax, Ay, Az] = operators;

        constexpr double k = 1.; // diffusivity
        constexpr double s = 1.; // source

        // -k * div q = s
        Ax(0, 1) = -k;
        Ay(0, 2) = -k;
        Az(0, 3) = -k;
        rhs[0]   = s;

        // grad T = q
        A0(1, 1) = -1.;
        Ax(1, 0) = 1.;
        A0(2, 2) = -1.;
        Ay(2, 0) = 1.;
        A0(3, 3) = -1.;
        Az(3, 0) = 1.;

        // rot q = 0
        Ay(4, 3) = 1.;
        Az(4, 2) = -1.;
        Ax(5, 3) = -1.;
        Az(5, 1) = 1.;
        Ax(6, 2) = 1.;
        Ay(6, 1) = -1.;
    });
    constexpr auto err_params         = KernelParams{.dimension = 3, .n_equations = 4, .n_fields = 4};
    constexpr auto error_kernel       = wrapDomainResidualKernel< err_params >([](const auto& in, auto& error) {
        const auto& [vals, ders, point]      = in;
        const auto& [T, qx, qy, qz]          = vals;
        const auto& [x_ders, y_ders, z_ders] = ders;

        const auto& q_xx = x_ders[1];
        const auto& q_yy = y_ders[2];
        const auto& q_zz = z_ders[3];
        const auto& T_x  = x_ders[0];
        const auto& T_y  = y_ders[0];
        const auto& T_z  = z_ders[0];

        constexpr double k = 1.;
        constexpr double s = 1.;

        error[0] = k * (q_xx + q_yy + q_zz) + s;
        error[1] = T_x - qx;
        error[2] = T_y - qy;
        error[3] = T_z - qz;
        return error;
    });
    constexpr auto dbc_params         = KernelParams{.dimension = 3, .n_equations = 1};
    constexpr auto dirichlet_bc_kernel =
        wrapBoundaryResidualKernel< dbc_params >([](const auto&, auto& out) { out[0] = 0.; });

    constexpr auto alg_params    = AlgebraicSystemParams{.eval_strategy = S, .cond_policy = CP};
    constexpr auto algpar_ctwrpr = L3STER_WRAP_CTVAL(alg_params);
    auto           alg_system    = makeAlgebraicSystem(comm, my_partition, probdef_ctwrpr, bc_def, algpar_ctwrpr);
    alg_system.beginAssembly();
    alg_system.assembleProblem(diffusion_kernel3d, {domain_id});
    alg_system.setDirichletBCValues(dirichlet_bc_kernel, boundary_ids, T_inds);
    alg_system.endAssembly();
    alg_system.describe();

    constexpr auto solver_opts  = IterSolverOpts{.verbosity = {.summary = true, .timing = true}};
    constexpr auto precond_opts = NativeJacobiOpts{};
    auto           solver       = CG{solver_opts, precond_opts};
    alg_system.solve(solver);

    L3STER_PROFILE_REGION_BEGIN("Solution management");
    auto solution_manager = SolutionManager{*my_partition, problem_def.n_fields};
    alg_system.updateSolution(dof_inds, solution_manager, field_inds);
    L3STER_PROFILE_REGION_END("Solution management");

    L3STER_PROFILE_REGION_BEGIN("Compute solution error");
    const auto field_access = solution_manager.makeFieldValueGetter(field_inds);
    const auto error        = computeNormL2(*comm, error_kernel, *my_partition, {domain_id}, field_access);
    L3STER_PROFILE_REGION_END("Compute solution error");

    if (comm->getRank() == 0)
        std::cout << std::format("\nThe L2 error components are:\n  {:.3e}\n  {:.3e}\n  {:.3e}\n  {:.3e}\n\n",
                                 error[0],
                                 error[1],
                                 error[2],
                                 error[3]);

    L3STER_PROFILE_REGION_BEGIN("Export results to VTK");
    auto           exporter    = PvtuExporter{comm, *my_partition};
    const auto     export_inds = util::gatherAsCommon(T_inds, T_grad_inds);
    constexpr auto field_names = std::array{"T"sv, "gradT"sv};
    exporter.exportSolution("Cube_Diffusion.pvtu", solution_manager, field_names, export_inds);
    exporter.flushWriteQueue();
    L3STER_PROFILE_REGION_END("Export results to VTK");
}