#include "l3ster/l3ster.hpp"

template < typename T, size_t N1, size_t N2 >
auto concatArrays(const std::array< T, N1 >& a1, const std::array< T, N2 >& a2) -> std::array< T, N1 + N2 >
{
    auto       retval = std::array< T, N1 + N2 >{};
    const auto out_it = std::ranges::copy(a1, retval.begin()).out;
    std::ranges::copy(a2, out_it);
    return retval;
}

int main(int argc, char* argv[])
{
    using namespace lstr;

    const auto scope_guard = L3sterScopeGuard{argc, argv};
    const auto comm        = MpiComm{MPI_COMM_WORLD};

    // Physical region IDs
    constexpr int domain = 13, inlet = 10, wall = 11, outlet = 12;

    // Number of unknown scalar fields: u, v, vorticity, p
    constexpr int IU = 0, IV = 1, IO = 2, IP = 3;
    constexpr int n_unknowns = 4;

    // Define the flow problem
    constexpr auto problem_def = ProblemDef{defineDomain< n_unknowns >(domain, ALL_DOFS)};

    // Dirichlet conditions velocity prescribed at inlet + walls
    constexpr auto dirichlet_def =
        ProblemDef{defineDomain< n_unknowns >(inlet, 0, 1), defineDomain< n_unknowns >(wall, 0, 1)};

    // Wrap as compile-time values
    constexpr auto probdef_ctwrpr = L3STER_WRAP_CTVAL(problem_def);
    constexpr auto dirdef_ctwrpr  = L3STER_WRAP_CTVAL(dirichlet_def);

    // Read mesh
    const std::string mesh_file  = argc > 1 ? argv[1] : "../karman.msh";
    constexpr el_o_t  mesh_order = 4; // el_o_t is the element order type
    const auto        mesh =
        readAndDistributeMesh(comm,
                              mesh_file,
                              mesh::gmsh_tag,
                              {inlet, wall, outlet} /* IDs of boundaries, needed for correct partitioning */,
                              L3STER_WRAP_CTVAL(mesh_order),
                              probdef_ctwrpr);

    // Algebraic system used for both the steady and unsteady problems
    auto algebraic_system = makeAlgebraicSystem(comm, mesh, probdef_ctwrpr, dirdef_ctwrpr);
    algebraic_system.describe(comm);

    // Time step
    constexpr double dt = .1;

    // Reynolds number
    constexpr double Re = 200.;
    constexpr double nu = /* cylinder diameter */ .8 * /* mean inlet velocity */ 1. / Re;

    // Kernels //
    // Steady state part of the Navier-Stokes kernel, defined here separately for reuse in the transient kernel
    constexpr auto fill_steady_kernel = [](auto&  A0,
                                           auto&  A1,
                                           auto&  A2,
                                           auto&  rhs,
                                           double u,
                                           double v,
                                           double du_dx,
                                           double dv_dx,
                                           double du_dy,
                                           double dv_dy) {
        // Momentum equations
        A0(0, IU) = 1. / dt + du_dx;
        A0(0, IV) = du_dy;
        A1(0, IU) = u;
        A1(0, IP) = 1.;
        A2(0, IV) = v;
        A2(0, IO) = nu;
        rhs(0, 0) = u * du_dx + v * du_dy;

        A0(1, IU) = dv_dx;
        A0(1, IV) = 1. / dt + dv_dy;
        A1(1, IV) = u;
        A1(1, IO) = -nu;
        A2(1, IV) = v;
        A2(1, IP) = 1.;
        rhs(1, 0) = u * dv_dx + v * dv_dy;

        // Incompressibility
        A1(2, IU) = 1.;
        A2(2, IV) = 1.;

        // Vorticity definition
        A0(3, IO) = 1.;
        A1(3, IV) = -1.;
        A2(3, IU) = 1.;
    };

    // Navier-Stokes kernel options (this is a non-linear equation, so we need to use a higher quadrature order)
    constexpr auto asm_opts     = AssemblyOptions{.value_order = 1, .derivative_order = 1};
    constexpr auto asmopt_ctval = L3STER_WRAP_CTVAL(asm_opts);

    // Steady state
    constexpr auto kernel_params_steady =
        KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = n_unknowns, .n_fields = 2};
    constexpr auto kernel_steady = wrapDomainEquationKernel< kernel_params_steady >([&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto& [u, v]                          = field_vals; // Velocity values from previous time steps
        const auto& [x_ders, y_ders]                = field_ders; // Partial velocity derivatives
        const auto& [du_dx, dv_dx]                  = x_ders;
        const auto& [du_dy, dv_dy]                  = y_ders;

        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;

        fill_steady_kernel(A0, A1, A2, rhs, u, v, du_dx, dv_dx, du_dy, dv_dy);
    });

    // Transient
    constexpr auto kernel_params_trans =
        KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = n_unknowns, .n_fields = 4};
    constexpr auto kernel_trans = wrapDomainEquationKernel< kernel_params_trans >([&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point]  = in;
        const auto& [u1, v1, u2, v2]                 = field_vals; // Velocity values from 2 previous iteration
        const auto& [x_ders, y_ders]                 = field_ders; // Partial velocity derivatives
        const auto& [du1_dx, dv1_dx, du2_dx, dv2_dx] = x_ders;
        const auto& [du1_dy, dv1_dy, du2_dy, dv2_dy] = y_ders;

        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;

        // Velocity extrapolations in time
        const double u     = 2 * u1 - u2;
        const double v     = 2 * v1 - v2;
        const double du_dx = 2 * du1_dx - du2_dx;
        const double dv_dx = 2 * dv1_dx - dv2_dx;
        const double du_dy = 2 * du1_dy - du2_dy;
        const double dv_dy = 2 * dv1_dy - dv2_dy;

        fill_steady_kernel(A0, A1, A2, rhs, u, v, du_dx, dv_dx, du_dy, dv_dy);

        // Add the time derivative contributions
        A0(0, IU) += 1.5 / dt;
        A0(1, IV) += 1.5 / dt;
        rhs(0, 0) += (2 * u1 - .5 * u2) / dt;
        rhs(1, 0) += (2 * v1 - .5 * v2) / dt;
    });

    // Outlet BC - this kernel operates only on the velocity components and the pressure
    constexpr auto outlet_dofs          = std::array< size_t, 3 >{IU, IV, IP};
    constexpr auto outdof_ctval         = L3STER_WRAP_CTVAL(outlet_dofs);
    constexpr auto kernel_params_outlet = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3};
    constexpr auto kernel_outlet = wrapBoundaryEquationKernel< kernel_params_outlet >([&](const auto& in, auto& out) {
        const double nx        = in.normal[0];
        const double ny        = in.normal[1];
        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;

        A0(0, 2) = -nx;
        A1(0, 0) = nu * nx;
        A2(0, 0) = nu * ny;

        A0(1, 2) = -ny;
        A1(1, 1) = nu * nx;
        A2(1, 1) = nu * ny;
    });

    // Inlet BC - parabolic velocity profile
    constexpr auto kernel_params_inlet = KernelParams{.dimension = 2, .n_equations = 2};
    constexpr auto kernel_inlet = wrapBoundaryResidualKernel< kernel_params_inlet >([&](const auto& in, auto& out) {
        const double y        = in.point.space.y();
        const double y_scaled = (y + 1.) / 2.;
        out[0]                = 1.5 * (1. - y_scaled * y_scaled);
        out[1]                = 0.;
    });

    // Set Dirichlet BC values
    constexpr auto bc_inds = std::array{0, 1}; // indices of DOFs for which we wish to prescribe a Dirichlet BC
    constexpr auto u_wall  = std::array{1., 0.};
    algebraic_system.setDirichletBCValues(u_wall, {wall}, bc_inds);
    algebraic_system.setDirichletBCValues(kernel_inlet, {inlet}, bc_inds);

    // We store 2 velocity snapshots, each consisting of 2 velocity components + the current vorticity and pressure
    auto       solution_manager = SolutionManager{*mesh, 6};
    auto       vel_inds1        = std::array{0, 1}; // Velocities from previous time step
    auto       vel_inds2        = std::array{2, 3}; // Velocities from 2 time steps ago
    const auto vort_inds        = std::array{4};    // Vorticity
    const auto p_inds           = std::array{5};    // Pressure
    const auto vort_press_inds  = std::array{4, 5}; // Pressure

    // Solution vector for the algebraic system
    auto solution = algebraic_system.initSolution();

    // L3STER interface to KLU2 direct solver
    auto solver = solvers::KLU2{};

    // Newton iterations for the stead state solution
    for (int iter = 0; iter != 10; ++iter)
    {
        // Zero out system
        algebraic_system.beginAssembly();

        // Velocity getter
        const auto vel_getter = solution_manager.makeFieldValueGetter(vel_inds1);

        // Assemble problem based on the defined kernels
        algebraic_system.assembleProblem(kernel_steady, {domain}, vel_getter, {}, asmopt_ctval);
        algebraic_system.assembleProblem(kernel_outlet, {outlet}, {}, outdof_ctval);

        // Finalize assembly
        algebraic_system.endAssembly();

        // Solve
        algebraic_system.solve(solver, solution);

        // Place the computed values in the solution manager
        algebraic_system.updateSolution(solution, vel_inds1, solution_manager, vel_inds1);
    }

    // Set the 2 remaining velocity snapshots to the steady state solution
    algebraic_system.updateSolution(solution, vel_inds1, solution_manager, vel_inds2);

    // Paraview exporter object
    auto exporter = PvtuExporter{*mesh};

    // Export initial snapshot
    exporter.exportSolution("results/karman_0.pvtu",
                            comm,
                            solution_manager,
                            {"Velocity", "Vorticity", "Pressure"},
                            std::array{vel_inds1, vort_inds, p_inds});

    constexpr int time_steps = 200;
    for (int time_step = 1; time_step <= time_steps; ++time_step)
    {
        // Zero out system
        algebraic_system.beginAssembly();

        // Velocity getter
        const auto vel_inds   = concatArrays(vel_inds1, vel_inds2);
        const auto vel_getter = solution_manager.makeFieldValueGetter(vel_inds);

        // Assemble problem based on the defined kernels
        algebraic_system.assembleProblem(kernel_trans, {domain}, vel_getter, {}, asmopt_ctval);
        algebraic_system.assembleProblem(kernel_outlet, {outlet}, {}, outdof_ctval);

        // Finalize assembly
        algebraic_system.endAssembly();

        // Solve
        algebraic_system.solve(solver, solution);

        // Place the computed values in the solution manager
        algebraic_system.updateSolution(
            solution, std::views::iota(0, 4), solution_manager, concatArrays(vel_inds2, vort_press_inds));

        // Export snapshot
        const auto file_name = "results/karman_" + std::to_string(time_step) + ".pvtu";
        exporter.exportSolution(file_name,
                                comm,
                                solution_manager,
                                {"Velocity", "Vorticity", "Pressure"},
                                std::array{vel_inds2, vort_inds, p_inds});

        // Swap velocity indices to avoid moving the underlying data
        std::swap(vel_inds1, vel_inds2);
    }
}