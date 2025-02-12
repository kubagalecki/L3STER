#include "l3ster/post/VtkExport.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <numbers>

using namespace lstr;
using namespace lstr::algsys;
using namespace lstr::mesh;
using std::numbers::pi;
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
    const auto     my_partition =
        generateAndDistributeMesh< 2 >(*comm, [&] { return mesh::makeSquareMesh(node_distx, node_disty); });

    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2;
    constexpr auto   params           = KernelParams{.dimension = 2, .n_equations = 2};
    const auto       bot_top_kernel   = wrapBoundaryResidualKernel< params >([&](const auto&, auto& out) {
        // Some arbitrary constant values
        out[0] = 1.;
        out[1] = pi;
    });
    const auto       kovasznay_kernel = wrapDomainResidualKernel< params >([&](const auto& in, auto& out) {
        // Kovasznay flow velocity field
        const double Re     = 40.;
        const double lambda = Re / 2. - std::sqrt(Re * Re / 4. - 4. * pi * pi);
        const auto&  p      = in.point.space;
        out[0]              = 1. - std::exp(lambda * p.x()) * std::cos(2. * pi * p.y());
        out[1]              = lambda * std::exp(lambda * p.x()) * std::sin(2 * pi * p.y()) / (2. * pi);
    });

    constexpr auto scalar_inds      = std::array{0, 2};
    constexpr auto vec_inds         = std::array{1, 3};
    auto           solution_manager = SolutionManager{*my_partition, scalar_inds.size() + vec_inds.size()};
    solution_manager.setFields(*comm, *my_partition, kovasznay_kernel, {domain_id}, vec_inds);
    solution_manager.setFields(*comm, *my_partition, bot_top_kernel, {bot_boundary, top_boundary}, scalar_inds);
    auto exporter   = PvtuExporter{comm, *my_partition};
    auto export_def = ExportDefinition{"2D/results"};
    export_def.defineField("C1", {scalar_inds.front()});
    export_def.defineField("Cpi", {scalar_inds.back()});
    export_def.defineField("vel", {vec_inds});
    exporter.exportSolution(export_def, solution_manager);
}

void test3D(const std::shared_ptr< MpiComm >& comm)
{
    const auto node_dist = [] {
        auto retval = std::array< double, 7 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -1. + static_cast< double >(i) * 2. / (retval.size() - 1);
        return retval;
    }();
    const auto my_partition = generateAndDistributeMesh< 2 >(*comm, [&] { return makeCubeMesh(node_dist); });

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

    constexpr d_id_t domain_id           = 0;
    constexpr auto   domain_field_inds   = std::array< size_t, 3 >{0, 1, 2};
    constexpr auto   boundary_field_inds = std::array< size_t, 3 >{3, 4, 5};
    auto solution_manager = SolutionManager{*my_partition, domain_field_inds.size() + domain_field_inds.size()};
    solution_manager.setFields(*comm, *my_partition, dom_kernel, {domain_id}, domain_field_inds);
    solution_manager.setFields(*comm, *my_partition, bnd_kernel, boundary_ids, boundary_field_inds);
    auto exporter   = PvtuExporter{comm, *my_partition};
    auto export_def = ExportDefinition{"3D/results.nonsense_extension"};
    export_def.defineField("vec3D", domain_field_inds);
    export_def.defineField("normal", boundary_field_inds);
    exporter.exportSolution(export_def, solution_manager);
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