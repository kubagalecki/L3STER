#include "l3ster/l3ster.hpp"

using namespace lstr;
// We will also be using some simple utilities from lstr::util, however these are not mandatory

// Channel dimensions
constexpr double L = 2.;
constexpr double W = 1.;

// Make a rectangular mesh, distribute it among the ranks of the passed MPI communicator
auto makeMesh(const MpiComm& comm)
{
    // Mesh parameters
    constexpr el_o_t mesh_order  = 6;
    const auto       node_dist_x = util::linspaceArray< 21 >(-1., 1.); // linspace semantics same as numpy
    const auto       node_dist_y = util::linspaceArray< 4 >(0., W);    // linspace semantics same as numpy

    const auto mesh_generator = [&] {
        return mesh::makeSquareMesh(node_dist_x, node_dist_y);
    };
    return generateAndDistributeMesh(comm,                           // mesh is distributed within this communicator
                                     mesh_generator,                 // mesh generator based on L3STER primitive
                                     L3STER_WRAP_CTVAL(mesh_order)); // mesh order passed as compile-time value,
}

int main(int argc, char* argv[])
{
    const auto scope_guard = L3sterScopeGuard{argc, argv};
    const auto comm        = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    // Assign names to physical domain IDs for readability
    constexpr int domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;

    // std::shared_ptr to the current rank's mesh partition
    const auto mesh = makeMesh(*comm);

    // Define the advection problem - 1 unknown (scalar concentration, at index 0) in the entire domain
    const auto problem_def = ProblemDefinition< 1 >{{domain_id}};

    // Left and right boundaries are periodic - the translation between them is [L, 0, 0]
    auto bc_def = BCDefinition< 1 >{};
    bc_def.definePeriodic({left_boundary}, {right_boundary}, {L, 0., 0.});

    // Algebraic system options - enable static condensation of internal element nodes
    constexpr auto algsys_opts = AlgebraicSystemParams{.eval_strategy = OperatorEvaluationStrategy::MatrixFree};

    // Algebraic system which we will need to fill - note opts struct is wrapped as compile-time value
    auto algebraic_system = makeAlgebraicSystem(comm, mesh, problem_def, bc_def, L3STER_WRAP_CTVAL(algsys_opts));
    algebraic_system.describe();

    // BDF 3
    constexpr auto time_order       = 3;
    constexpr auto bdf_leading_coef = 11. / 6.;
    constexpr auto bdf_coefs        = std::array< double, time_order >{3., -1.5, 1. / 3.};

    // Time step
    constexpr double dt = .02;

    // Advection velocity
    constexpr double u = 1., v = 0.;

    // Kernel parameters - we need to pass 3 fields (values of phi at the 3 previous time steps) to the kernel
    constexpr auto adv_params = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1, .n_fields = time_order};

    // Define the advection equation kernel
    const auto advection_kernel = wrapDomainEquationKernel< adv_params >([&](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        auto& [operators, rhs]                      = out;
        auto& [A0, A1, A2]                          = operators;

        A0(0, 0) = bdf_leading_coef;
        A1(0, 0) = u * dt;
        A2(0, 0) = v * dt;

        // Field vals is a std::array of size kernel_params.n_fields - we can use STL algorithms
        rhs[0] = std::transform_reduce(field_vals.begin(), field_vals.end(), bdf_coefs.begin(), 0.); // inner product
    });

    // Define the Neumann BC kernel
    constexpr auto neu_params     = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 1};
    const auto     neumann_kernel = wrapBoundaryEquationKernel< neu_params >([&](const auto& in, auto& out) {
        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;

        A1(0, 0) = in.normal[0];
        A2(0, 0) = in.normal[1];
    });

    // Analytical solution - used for setting the initial values and computing the error
    const auto analytical_solution = [&](const auto& in, auto& out) {
        const auto t    = in.point.time;
        const auto x    = in.point.space.x();
        const auto dv   = t * u;
        auto       x_dv = x - dv;
        while (x_dv < -1.) // include domain periodicity
            x_dv += L;

        // Gauss curve
        out[0] = std::exp(-10. * x_dv * x_dv);
    };

    // Parameters for solution kernel
    constexpr auto solution_params = KernelParams{.dimension = 2, .n_equations = 1};

    // Solution *residual* kernel - only defines the RHS
    const auto solution_kernel = wrapDomainResidualKernel< solution_params >(analytical_solution);

    // Parameters for error kernel - 1 field value representing the current solution
    constexpr auto error_params = KernelParams{.dimension = 2, .n_equations = 1, .n_fields = 1};

    // Error kernel := current value - analytical solution
    const auto error_kernel = wrapDomainResidualKernel< error_params >([&](const auto& in, auto& out) {
        const auto solution = in.field_vals[0];
        analytical_solution(in, out); // Reuse this function
        out[0] = solution - out[0];
    });

    // Integral of the solution over the domain, used for normalizing the error
    const auto sol_integral = computeIntegral(*comm, solution_kernel, *mesh, {domain_id})[0];

    // We need to store 3 solution snapshots in the solution manager
    auto solution_manager = SolutionManager{*mesh, 3};

    // Array of indices of time steps, starting from the most recent. This array will be rotated after every time step,
    // so that the oldest time step index becomes the second oldest, etc. These indices will be used to index into the
    // slots of the solution manager.
    auto time_hist_inds = util::makeIotaArray< unsigned, time_order >();

    // Set the solution history using the analytical solution kernel. In real applications, where you don't know the
    // history, you can gradually ramp up the time stepping order.
    for (auto i : time_hist_inds)
        solution_manager.setFields(*comm, *mesh, solution_kernel, {domain_id}, {i}, {}, -dt * i);

    // L3STER-native Jacobi preconditioner options
    constexpr auto precond_opts = NativeJacobiOpts{};

    // Iterative solver options - defaults are usually ok, here we change them for demonstration purposes
    constexpr auto solver_opts = IterSolverOpts{
        .tol = 1.e-4, .max_iters = 1000, .verbosity = {.summary = true, .iter_details = true, .timing = true}};

    // Conjugate gradients solver - the preconditioner type is deduced based on the type of precond_opts
    auto solver = CG{solver_opts, precond_opts};

    // Paraview exporter object
    auto exporter = PvtuExporter{comm, *mesh};

    // Export initial solution
    auto export_def = ExportDefinition{"results/phi_000.pvtu"};
    export_def.defineField("phi", {0});
    exporter.exportSolution(export_def, solution_manager);

    // Print error report header
    std::cout << std::format(
        "|{:^11}|{:^16}|{:^19}|{:^17}|\n", "Time Step", "Rel. Error [%]", "CG solution error", "# CG iterations");

    const auto num_steps = static_cast< unsigned >(std::lround(L / dt)); // 1 period
    for (unsigned time_step = 1; time_step <= num_steps; ++time_step)
    {
        // Current time
        const auto time = time_step * dt;

        // Zero out system
        algebraic_system.beginAssembly();

        // Access time history according to the current state of the index array
        const auto field_access = solution_manager.getFieldAccess(time_hist_inds);

        // Assemble problem based on the defined kernels
        // The operator is matrix-free, so no actual computation takes place here
        algebraic_system.assembleProblem(advection_kernel, {domain_id}, field_access);
        algebraic_system.assembleProblem(neumann_kernel, {bot_boundary, top_boundary});

        // Finalize assembly
        algebraic_system.endAssembly();

        // Solve - iterative solver returns some convergence information
        const auto [tol, iters] = algebraic_system.solve(solver);

        // Index of the oldest time step - this will now be overwritten
        const auto last_ind = time_hist_inds.back();

        // Place the computed value of phi in the solution manager at the last index
        algebraic_system.updateSolution({0}, solution_manager, {last_ind});

        // Export snapshot
        export_def = ExportDefinition{std::format("results/phi_{:03}.pvtu", time_step)};
        export_def.defineField("phi", {last_ind});
        exporter.exportSolution(export_def, solution_manager);

        // Compute the L2 error norm
        const auto error_access = solution_manager.getFieldAccess(std::array{last_ind});
        const auto error        = computeNormL2(*comm, error_kernel, *mesh, {domain_id}, error_access, {}, time)[0];
        std::cout << std::format(
            "|{:^11}|{:^16.3f}|{:^19.2e}|{:^17}|\n", time_step, 100. * error / (sol_integral * W * L), tol, iters);

        // Left-rotate time history indices - last_ind is moved to the front of the array
        std::ranges::rotate(time_hist_inds, std::prev(time_hist_inds.end()));
    }
}