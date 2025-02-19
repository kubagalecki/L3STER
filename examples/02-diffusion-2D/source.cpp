#include "l3ster/l3ster.hpp"

using namespace lstr;
// We will also be using some simple utilities from lstr::util, however these are not mandatory

// Make a unit square mesh, distribute it among the ranks of the passed MPI communicator
auto makeMesh(const MpiComm& comm)
{
    // Mesh parameters
    constexpr el_o_t mesh_order = 4;
    const auto       node_dist  = util::linspaceArray< 21 >(0., 1.); // linspace semantics same as numpy

    const auto mesh_generator = [&] {
        return mesh::makeSquareMesh(node_dist);
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
    constexpr int  domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    constexpr auto boundary_ids = std::array{bot_boundary, top_boundary, left_boundary, right_boundary};

    // std::shared_ptr to the current rank's mesh partition
    const auto mesh = makeMesh(*comm);

    // Define the diffusion problem - 3 unknowns (phi + qx ,qy) in the entire domain
    const auto problem_def = ProblemDefinition< 3 >{{domain_id}};

    // Algebraic system which we will need to fill
    auto algebraic_system = makeAlgebraicSystem(comm, mesh, problem_def);
    algebraic_system.describe();

    // Kernel params - these have to be known at compile time
    constexpr auto domain_kernel_params   = KernelParams{.dimension = 2, .n_equations = 4, .n_unknowns = 3};
    constexpr auto boundary_kernel_params = KernelParams{.dimension = 2, .n_equations = 1, .n_unknowns = 3};

    // Kernel definition - in general this does not need to be constexpr (you can capture by reference)
    constexpr auto domain_kernel = wrapDomainEquationKernel< domain_kernel_params >([](const auto&, auto& out) {
        // Note that we don't use the first argument (the kernel input), since the kernel is constant in the domain
        // The unknowns are ordered as follows: phi, qx, qy (this is an arbitrary assumption)

        // These are already initialized to 0, we only need to fill in the non-zero entries
        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;

        // -div q = 1
        A1(0, 1) = -1.;
        A2(0, 2) = -1.;
        rhs[0]   = 1.;

        // grad phi - q = 0
        A0(1, 1) = -1.;
        A1(1, 0) = 1.;
        A0(2, 2) = -1.;
        A2(2, 0) = 1.;

        // curl q = 0 (only 1 equation since we're in 2D)
        A1(3, 2) = 1.;
        A2(3, 1) = -1.;
    });
    constexpr auto bc_kernel = wrapBoundaryEquationKernel< boundary_kernel_params >([](const auto& in, auto& out) {
        // Unit boundary normal (obviously this is only available in boundary kernels)
        const auto& normal = in.normal;
        const auto  nx     = normal[0];
        const auto  ny     = normal[1];

        auto& [operators, rhs] = out;
        auto& [A0, A1, A2]     = operators;

        // phi + q * n = 0
        A0(0, 0) = 1.;
        A0(0, 1) = nx;
        A0(0, 2) = ny;
    });

    // Zero out the system
    algebraic_system.beginAssembly();

    // Diffusion equation
    algebraic_system.assembleProblem(domain_kernel, {domain_id});

    // Boundary condition
    algebraic_system.assembleProblem(bc_kernel, boundary_ids);

    // Finalize assembly
    algebraic_system.endAssembly();

    // L3STER interface to the KLU2 direct solver
    auto solver = Klu2{};

    // Solve the algebraic system
    algebraic_system.solve(solver);

    // The algebraic system stores the computed solution, but we can't use it directly in a useful way.
    // The solution manager allows us to store computed fields for later access. We define how many "slots" it has -
    // each slot can hold to a single scalar field
    auto solution_manager = SolutionManager{*mesh, /* number of slots - we only need 1 for phi */ 1};

    // Place the value of the first solution component (phi) in the first (and only) slot of the solution manager
    // Here we're not interested in q, but in general we could export its components as well
    algebraic_system.updateSolution({0}, solution_manager, {0});

    // Paraview exporter object
    auto exporter = PvtuExporter{comm, *mesh};

    // Export phi from the solution manager to a .pvtu file for viewing in Paraview
    auto export_def = ExportDefinition{"results/phi.pvtu"}; // Path to file
    export_def.defineField("phi", {0});                     // Field name, index of solution manager slot
    exporter.exportSolution(export_def, solution_manager);  // Write to file
}