#include "Common.hpp"

#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/glob_asm/AlgebraicSystem.hpp"
#include "l3ster/glob_asm/ComputeValuesAtNodes.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/VtkExport.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <numbers>

using namespace lstr;
using namespace lstr::glob_asm;
using namespace lstr::mesh;
using namespace std::numbers;
using namespace std::string_view_literals;

void test2D(const std::shared_ptr< MpiComm >& comm)
{
    constexpr auto node_distx = std::invoke([] {
        auto retval = std::array< double, 11 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + static_cast< double >(i) * 2.5 / (retval.size() - 1);
        return retval;
    });
    constexpr auto node_disty = std::invoke([] {
        auto retval = std::array< double, 15 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + static_cast< double >(i) * 2. / (retval.size() - 1);
        return retval;
    });
    constexpr auto mesh_order = 2;
    const auto     my_partition =
        generateAndDistributeMesh< mesh_order >(*comm, [&] { return mesh::makeSquareMesh(node_distx, node_disty); });

    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2;
    constexpr auto   problem_def       = ProblemDef{defineDomain< 4 >(domain_id, 1, 3),
                                            defineDomain< 4 >(bot_boundary, 0, 2),
                                            defineDomain< 4 >(top_boundary, 0, 2)};
    constexpr auto   problemdef_ctwrpr = util::ConstexprValue< problem_def >{};
    constexpr auto   scalar_inds       = std::array< size_t, 2 >{0, 2};
    constexpr auto   vec_inds          = std::array< size_t, 2 >{1, 3};
    constexpr auto   all_field_inds    = util::makeIotaArray< size_t, problem_def.n_fields >();

    auto system_manager = makeAlgebraicSystem(comm, my_partition, problemdef_ctwrpr);
    system_manager.endAssembly();
    auto solution_manager = SolutionManager{*my_partition, problem_def.n_fields};

    constexpr auto params           = KernelParams{.dimension = 2, .n_equations = 2};
    const auto     bot_top_kernel   = wrapBoundaryResidualKernel< params >([&](const auto&, auto& out) {
        // Some arbitrary constant values
        out[0] = 1.;
        out[1] = pi;
    });
    const auto     kovasznay_kernel = wrapDomainResidualKernel< params >([&](const auto& in, auto& out) {
        // Kovasznay flow velocity field
        const double Re     = 40.;
        const double lambda = Re / 2. - std::sqrt(Re * Re / 4. - 4. * pi * pi);
        const auto&  p      = in.point.space;
        out[0]              = 1. - std::exp(lambda * p.x()) * std::cos(2. * pi * p.y());
        out[1]              = lambda * std::exp(lambda * p.x()) * std::sin(2 * pi * p.y()) / (2. * pi);
    });
    system_manager.setValues(system_manager.getSolution(), kovasznay_kernel, {domain_id}, vec_inds);
    system_manager.setValues(system_manager.getSolution(), bot_top_kernel, {bot_boundary, top_boundary}, scalar_inds);
    system_manager.updateSolution(all_field_inds, solution_manager, all_field_inds);

    auto       exporter        = PvtuExporter{comm, *my_partition};
    const auto field_names     = std::array{"C1"sv, "Cpi"sv, "vel"sv};
    const auto field_comp_inds = std::array< std::span< const size_t >, 3 >{
        std::span{std::addressof(scalar_inds[0]), 1}, std::span{std::addressof(scalar_inds[1]), 1}, vec_inds};
    comm->barrier();
    exporter.exportSolution("2D/results", solution_manager, field_names, field_comp_inds);
}

void test3D(const std::shared_ptr< MpiComm >& comm)
{
    const auto node_dist = [] {
        auto retval = std::array< double, 7 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -1. + static_cast< double >(i) * 2. / (retval.size() - 1);
        return retval;
    }();
    constexpr auto mesh_order = 2;
    const auto my_partition   = generateAndDistributeMesh< mesh_order >(*comm, [&] { return makeCubeMesh(node_dist); });

    constexpr d_id_t domain_id           = 0;
    constexpr auto   problem_def         = ProblemDef{defineDomain< 6 >(domain_id, 0, 1, 2),
                                            defineDomain< 6 >(1, 3, 4, 5),
                                            defineDomain< 6 >(2, 3, 4, 5),
                                            defineDomain< 6 >(3, 3, 4, 5),
                                            defineDomain< 6 >(4, 3, 4, 5),
                                            defineDomain< 6 >(5, 3, 4, 5),
                                            defineDomain< 6 >(6, 3, 4, 5)};
    constexpr auto   problemdef_ctwrpr   = util::ConstexprValue< problem_def >{};
    constexpr auto   n_fields            = problem_def.n_fields;
    constexpr auto   domain_field_inds   = std::array< size_t, 3 >{0, 1, 2};
    constexpr auto   boundary_field_inds = std::array< size_t, 3 >{3, 4, 5};
    constexpr auto   field_inds          = util::makeIotaArray< size_t, n_fields >();

    auto system_manager = makeAlgebraicSystem(comm, my_partition, problemdef_ctwrpr);
    system_manager.endAssembly();
    auto solution_manager = SolutionManager{*my_partition, n_fields};

    constexpr auto ker_params   = KernelParams{.dimension = 3, .n_equations = 3};
    const auto     dom_kernel   = wrapDomainResidualKernel< ker_params >([&](const auto& in, auto& out) {
        const auto& p = in.point.space;
        const auto  r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
        out[0]        = r;
        out[1]        = p.y();
        out[2]        = p.z();
    });
    constexpr auto boundary_ids = util::makeIotaArray< d_id_t, 6 >(1);
    const auto     bnd_kernel =
        wrapBoundaryResidualKernel< ker_params >([](const auto& in, auto& out) { out = in.normal; });
    system_manager.setValues(system_manager.getSolution(), bnd_kernel, boundary_ids, boundary_field_inds);
    system_manager.setValues(
        system_manager.getSolution(), dom_kernel, std::views::single(domain_id), domain_field_inds);
    system_manager.updateSolution(field_inds, solution_manager, field_inds);

    auto       exporter    = PvtuExporter{comm, *my_partition};
    const auto field_names = std::array{"vec3D"sv, "normal"sv};
    comm->barrier();
    exporter.exportSolution("3D/results.nonsense_extension",
                            solution_manager,
                            field_names,
                            std::array{domain_field_inds, boundary_field_inds});
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    const auto comm          = std::make_shared< MpiComm >(MPI_COMM_WORLD);
    test2D(comm);
    test3D(comm);
    // TODO: Programatically check whether the data was exported correctly. For the time being, check manually...
}