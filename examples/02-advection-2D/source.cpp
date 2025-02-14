#include "l3ster/l3ster.hpp"

// Channel dimensions
constexpr double L = 3.;
constexpr double w = 1.;

using namespace lstr;
// We will also be using some simple utilities from lstr::util, however these are not mandatory

// Make a rectangular mesh, distribute it among the ranks of the passed MPI communicator
auto makeMesh(const MpiComm& comm, const auto& problem_def)
{
    // Mesh parameters
    constexpr el_o_t mesh_order  = 4;
    const auto       node_dist_x = util::linspaceArray< 61 >(0., L); // linspace semantics same as numpy
    const auto       node_dist_y = util::linspaceArray< 21 >(0., w); // linspace semantics same as numpy

    const auto mesh_generator = [&] {
        return mesh::makeSquareMesh(node_dist_x, node_dist_y);
    };
    return generateAndDistributeMesh(comm,                          // mesh is distributed within this communicator
                                     mesh_generator,                // mesh generator based on L3STER primitive
                                     L3STER_WRAP_CTVAL(mesh_order), // mesh order passed as compile-time value,
                                     problem_def);                  // (optional) problem def helps with load-balancing
}

int main(int argc, char* argv[])
{
    const auto scope_guard = L3sterScopeGuard{argc, argv};
    const auto comm        = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    // Assign names to physical domain IDs for readability
    constexpr int domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;

    // Define the advection problem - 1 unknown (scalar concentration, at index 0) in the entire domain
    const auto problem_def = ProblemDefinition< 1 >{{domain_id}};

    // Dirichlet condition - we will be setting the value of the scalar concentration at the left (inflow) boundary
    auto bc_def = BCDefinition< 1 >{};
    bc_def.defineDirichlet({left_boundary});

    // std::shared_ptr to the current rank's mesh partition
    const auto mesh = makeMesh(*comm, problem_def);

    // Algebraic system which we will need to fill
    auto algebraic_system = makeAlgebraicSystem(comm, mesh, problem_def, bc_def);
    algebraic_system.describe();

    // Time step
    constexpr double dt = .1;

    // Kernel definition
    constexpr auto kernel_params = KernelParams{.dimension = 2, .n_equations = 1, .n_fields = 1};
    constexpr auto kernel        = wrapDomainEquationKernel< kernel_params >([](const auto& in, auto& out) {
        const auto& [field_vals, field_ders, point] = in;
        const auto   phi_prev                       = field_vals[0];             // phi at previous time step
        const double y_scaled                       = point.space.y() * 2. - 1.; // helper value
        const double vx = 1. - y_scaled * y_scaled; // x advection velocity - based on parabolic profile
        const double vy = 0.;                       // y advection velocity

        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;
        A0(0, 0)               = 1. / dt;
        A1(0, 0)               = vx;
        A2(0, 0)               = vy;
        rhs(0, 0)              = phi_prev / dt;
    });

    // Set Dirichlet BC values. Since these don't depend on time, we don't need to update them in the main loop
    constexpr auto bc_inds  = std::array{0};  // index of DOFs for which we wish to prescribe a Dirichlet BC
    constexpr auto bc_value = std::array{1.}; // value we wish to prescribe
    algebraic_system.setDirichletBCValues(bc_value, /*IDs of boundaries*/ {left_boundary}, bc_inds);

    // Solution manager which stores snapshots of solution components. Here we are using a 1st order time discretization
    // scheme, so we only store the concentration at the previous time step. The stored values are initialized to 0 by
    // default, which is our initial condition, so we don't need to do anything else.
    auto solution_manager = SolutionManager{*mesh, /* number of solution components to store */ 1};

    // Object used to access the values stored in the solution manager
    const auto phi_prev_ind    = std::array< size_t, 1 >{0}; // array of indices of components to access
    const auto phi_prev_getter = solution_manager.makeFieldValueGetter(phi_prev_ind);

    // L3STER interface to KLU2 direct solver
    auto solver = solvers::KLU2{};

    // Paraview exporter object
    auto exporter = PvtuExporter{comm, *mesh};

    // Subsequently used indices
    const auto dof_inds = std::array{0}; // Indices of DOFs in the solution vector
    const auto sol_inds = std::array{0}; // Target indices in the solution manager

    // Export initial snapshot (phi = 0)
    auto export_def_init = ExportDefinition{"results/phi_000.pvtu"};
    export_def_init.defineField("phi", {0});
    exporter.exportSolution(export_def_init, solution_manager);

    constexpr int time_steps = 20;
    for (int time_step = 1; time_step <= time_steps; ++time_step)
    {
        // Zero out system
        algebraic_system.beginAssembly();

        // Assemble problem based on the defined kernel
        algebraic_system.assembleProblem(kernel, {domain_id}, phi_prev_getter);

        // Finalize assembly
        algebraic_system.endAssembly();

        // Solve
        algebraic_system.solve(solver);

        // Place the computed values in the solution manager
        algebraic_system.updateSolution(dof_inds, solution_manager, sol_inds);

        // Export snapshot
        auto export_def = ExportDefinition{std::format("results/phi_{:03}.pvtu", time_step)};
        export_def.defineField("phi", {0});
        exporter.exportSolution(export_def, solution_manager);
    }
}