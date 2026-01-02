#include "Common.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CylinderInChannel2D.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/post/SaveLoad.hpp"
#include "l3ster/post/VtkExport.hpp"
#include "l3ster/util/ScopeGuards.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    const auto par_guard   = util::MaxParallelismGuard{4};
    const auto scope_guard = L3sterScopeGuard{argc, argv};
    auto       comm        = std::make_shared< MpiComm >(MPI_COMM_WORLD);

    const auto     mesh_opts  = mesh::CylinderInChannel2DGeometry{.n_circumf = 16, .n_radial = 5};
    constexpr auto mesh_order = 4;
    const auto     mesh =
        generateAndDistributeMesh< mesh_order >(*comm, [&] { return mesh::makeCylinderInChannel2DMesh(mesh_opts); });

    auto           solution_manager = SolutionManager{*mesh, 2};
    constexpr auto kernel_params    = KernelParams{.dimension = 2, .n_equations = 2};
    constexpr auto set_fields       = [](const auto& in, auto& out) {
        const auto x = in.point.space.x();
        const auto y = in.point.space.y();
        out[0]       = std::sin(.5 * x);
        out[1]       = std::cos(.5 * y);
    };
    const auto kernel = wrapDomainResidualKernel< kernel_params >(set_fields);
    solution_manager.setFields(*comm, *mesh, kernel, {0}, {0, 1});

    const auto mesh_path    = "mesh.l3s";
    const auto results_path = "results.l3s";
    save(*comm, *mesh, mesh_path);
    save(*comm, *mesh, solution_manager, results_path);

    auto       loader      = Loader< mesh_order >{mesh_path};
    const auto loaded_mesh = loader.loadMesh(*comm);
    solution_manager       = SolutionManager{*loaded_mesh, 2};
    loader.loadResults(results_path, solution_manager);

    constexpr auto err_kernel_params = KernelParams{.dimension = 2, .n_equations = 2, .n_fields = 2};
    const auto     error_kernel      = wrapDomainResidualKernel< err_kernel_params >([&](const auto& in, auto& out) {
        set_fields(in, out);
        const auto& [vals, ders, _] = in;
        out[0] -= vals[0];
        out[1] -= vals[1];
    });
    const auto     field_access      = solution_manager.getFieldAccess(std::array{0, 1});
    const auto     loaded_error      = computeNormL2(*comm, error_kernel, *loaded_mesh, {0}, field_access);
    REQUIRE(loaded_error.norm() < 1e-3);
}