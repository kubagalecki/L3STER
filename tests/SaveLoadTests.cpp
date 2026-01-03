#include "Common.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CylinderInChannel2D.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/post/SaveLoad.hpp"
#include "l3ster/util/ScopeGuards.hpp"

using namespace lstr;

constexpr auto mesh_order = 4;

constexpr auto set_fields = [](const auto& in, auto& out) {
    const auto x = in.point.space.x();
    const auto y = in.point.space.y();
    out[0]       = std::sin(.5 * x);
    out[1]       = std::cos(.5 * y);
};

auto makeMesh(const MpiComm& comm)
{
    using std::numbers::pi;
    constexpr auto mesh_opts = mesh::CylinderInChannel2DGeometry{.r_inner       = pi / 4.,
                                                                 .r_outer       = pi / 2.,
                                                                 .left_offset   = pi,
                                                                 .right_offset  = pi,
                                                                 .bottom_offset = pi,
                                                                 .top_offset    = pi,
                                                                 .n_circumf     = 16,
                                                                 .n_radial      = 5,
                                                                 .n_left        = 5,
                                                                 .n_right       = 5,
                                                                 .n_bottom      = 5,
                                                                 .n_top         = 5};
    return generateAndDistributeMesh< mesh_order >(comm, [&] { return mesh::makeCylinderInChannel2DMesh(mesh_opts); });
}

void saveResults(const MpiComm& comm, std::string_view file_name)
{
    const auto     mesh_ptr         = makeMesh(comm);
    auto           solution_manager = SolutionManager{*mesh_ptr, 2};
    constexpr auto kernel_params    = KernelParams{.dimension = 2, .n_equations = 2};
    const auto     kernel           = wrapDomainResidualKernel< kernel_params >(set_fields);
    solution_manager.setFields(comm, *mesh_ptr, kernel, {0}, {0, 1});

    const auto mesh_path    = std::string{file_name} + ".mesh";
    const auto results_path = std::string{file_name} + ".res";
    save(comm, *mesh_ptr, mesh_path);
    save(comm, *mesh_ptr, solution_manager, results_path);
}

auto load(const MpiComm& comm, std::string_view file_name, const Loader< mesh_order >::Opts& opts)
{
    const auto mesh_path        = std::string{file_name} + ".mesh";
    const auto results_path     = std::string{file_name} + ".res";
    auto       loader           = Loader< mesh_order >{mesh_path, opts};
    auto       loaded_mesh      = loader.loadMesh(comm);
    auto       solution_manager = SolutionManager{*loaded_mesh, 2};
    loader.loadResults(results_path, solution_manager);
    return std::make_pair(std::move(loaded_mesh), std::move(solution_manager));
}

void checkResults(const MpiComm& comm, std::string_view file_name, const Loader< mesh_order >::Opts& opts)
{
    comm.barrier();
    const auto [mesh, solution_manager] = load(comm, file_name, opts);
    constexpr auto err_kernel_params    = KernelParams{.dimension = 2, .n_equations = 2, .n_fields = 2};
    const auto     error_kernel         = wrapDomainResidualKernel< err_kernel_params >([&](const auto& in, auto& out) {
        set_fields(in, out);
        const auto& [vals, ders, _] = in;
        out[0] -= vals[0];
        out[1] -= vals[1];
    });
    const auto     field_access         = solution_manager.getFieldAccess(std::array{0, 1});
    const auto     loaded_error         = computeNormL2(comm, error_kernel, *mesh, {0}, field_access);
    const auto     error                = loaded_error.norm();
    constexpr auto threshold            = 1e-6;
    if (comm.getRank() == 0)
        error < threshold ? std::println("Error: {:.2e} < {:.1e}; PASS", error, threshold)
                          : std::println(stderr, "Error: {:.2e} >= {:.1e}; FAIL", error, threshold);
    REQUIRE(error < threshold);
}

int main(int argc, char* argv[])
{
    using namespace lstr;
    const auto par_guard   = util::MaxParallelismGuard{4};
    const auto scope_guard = L3sterScopeGuard{argc, argv};
    const auto comm_self   = MpiComm{MPI_COMM_SELF};
    const auto comm_world  = MpiComm{MPI_COMM_WORLD};

    saveResults(comm_self, "serial");
    saveResults(comm_world, "parallel");

    checkResults(comm_self, "parallel", {.repartition = true});
    checkResults(comm_world, "serial", {.repartition = true});
    checkResults(comm_world, "parallel", {.repartition = false});
    if (comm_world.getSize() > 1)
    {
        CHECK_THROWS(checkResults(comm_world, "serial", {.repartition = false}););
        CHECK_THROWS(checkResults(comm_self, "parallel", {.repartition = false}););
    }
}