#include "l3ster/l3ster.hpp"

#include <numbers>

using namespace lstr;

// Unit square mesh generation - see example #2
constexpr el_o_t mesh_order = 4;
auto             makeMesh(const MpiComm& comm)
{
    const auto node_dist = util::linspaceArray< 11 >(0., 1.);

    const auto mesh_generator = [&] {
        return mesh::makeSquareMesh(node_dist);
    };
    return generateAndDistributeMesh(comm, mesh_generator, L3STER_WRAP_CTVAL(mesh_order));
}

// Sample results to save
// 3 fields: cos(2*pi*x), sin(2*pi*y), x**2 + y**2
void exampleFields(const auto& in, auto& out)
{
    using std::numbers::pi;
    const auto& point = in.point.space;
    const auto  x     = point.x();
    const auto  y     = point.y();
    out[0]            = std::cos(2. * pi * x);
    out[1]            = std::sin(2. * pi * y);
    out[2]            = x * x + y * y;
}

// Initialize fields in the solution manager using a kernel - see example #4
auto makeExampleSolution(const MpiComm& comm, const auto& mesh) -> SolutionManager
{
    constexpr auto kernel_params = KernelParams{.dimension = 2, .n_equations = 3};
    constexpr auto kernel        = wrapDomainResidualKernel< kernel_params >([](const auto& in, auto& out) {
        using std::numbers::pi;
        const auto& point = in.point.space;
        const auto  x     = point.x();
        const auto  y     = point.y();
        out[0]            = std::cos(2. * pi * x);
        out[1]            = std::sin(2. * pi * y);
        out[2]            = x * x + y * y;
    });
    constexpr auto domain_ids    = std::array{0};
    constexpr auto solman_inds   = std::array{0, 1, 2};
    auto           retval        = SolutionManager{mesh, kernel_params.n_equations};
    retval.setFields(comm, mesh, kernel, domain_ids, solman_inds);
    return retval;
}

// File names used in this example, extensions are arbitrary
auto getFileNames() -> std::array< const char*, 2 >
{
    return {"example.mesh", "example.results"};
}

// Save mesh and results to separate files
void save()
{
    const auto comm             = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    const auto mesh             = makeMesh(*comm);
    const auto solution_manager = makeExampleSolution(*comm, *mesh);

    // File names
    const auto [mesh_fn, results_fn] = getFileNames();

    // Output file comments. These are placed in the header section (ASCII text) and can help with identifying the
    // files. Comments are optional.
    const auto mesh_comment    = "Mesh file";
    const auto results_comment = "Results";

    // Save mesh
    save(*comm, *mesh, mesh_fn, mesh_comment);

    // Save all results in the solution manager. You can also specify a subset and/or permutation of fields to be saved
    // by passing their indices. This can be useful during time-stepping, where the indices of the solution history
    // change over the course of the simulation.
    save(*comm, *mesh, solution_manager, results_fn, results_comment);
}

// Load results from files and verify they are correct
void loadAndVerify()
{
    // Create communicator. Note that the number of MPI processes for the save and load operations can differ - L3STER
    // supports this situation
    const auto comm = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    // File names
    const auto [mesh_fn, results_fn] = getFileNames();

    // Loader object
    auto loader = Loader< mesh_order >{mesh_fn};

    // Load mesh
    const auto mesh = loader.loadMesh(*comm);

    // Solution manager
    auto solution_manager = SolutionManager{*mesh, 3};

    // Load the results. You can also specify source and destination indices when loading a subset of the results, or
    // when a permutation is required.
    loader.loadResults(results_fn, solution_manager);

    // Check that the loaded results are equal to the example fields defined above
    constexpr auto kernel_params = KernelParams{.dimension = 2, .n_equations = 3, .n_fields = 3};
    constexpr auto error_kernel  = wrapDomainResidualKernel< kernel_params >([](const auto& in, auto& out) {
        // Initialize with expected solution
        exampleFields(in, out);

        // Subtract loaded solution
        for (int i = 0; i != 3; ++i)
            out[i] -= in.field_vals[i];
    });
    constexpr auto domain_ids = std::array{0};
    constexpr auto solman_inds   = std::array{0, 1, 2};
    const auto     field_access  = solution_manager.getFieldAccess(solman_inds);
    const auto     errors        = computeNormL2(*comm, error_kernel, *mesh, domain_ids, field_access);

    // Print L2 error. Note that trigonometric functions can't be represented exactly by polynomials - hence some error.
    if (comm->getRank() == 0)
        std::println("Loaded results errors: {:.3e}, {:.3e}, {:.3e}", errors[0], errors[1], errors[2]);
}

int main(int argc, char* argv[])
{
    const auto scope_guard = L3sterScopeGuard{argc, argv};

    save();
    loadAndVerify();
}