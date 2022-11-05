#include "l3ster/post/VtkExport.hpp"
#include "l3ster/assembly/AlgebraicSystemManager.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/assembly/SolutionManager.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include <numbers>

using namespace lstr;
using namespace std::numbers;
using namespace std::string_view_literals;

void vtkExportTest2D()
{
    const MpiComm comm;

    const auto node_distx = [] {
        std::array< double, 13 > retval;
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + i * 2.5 / (retval.size() - 1);
        return retval;
    }();
    const auto node_disty = [] {
        std::array< double, 21 > retval;
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + i * 2. / (retval.size() - 1);
        return retval;
    }();

    Mesh mesh;
    if (comm.getRank() == 0)
    {
        constexpr auto mesh_order = 2;
        mesh                      = makeSquareMesh(node_distx, node_disty);
        mesh.getPartitions()[0].initDualGraph();
        mesh.getPartitions()[0] = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    }
    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});

    constexpr auto problem_def       = std::array{Pair{domain_id, std::array{false, true, false, true}},
                                            Pair{bot_boundary, std::array{true, false, true, false}},
                                            Pair{top_boundary, std::array{true, false, true, false}}};
    constexpr auto problemdef_ctwrpr = ConstexprValue< problem_def >{};
    constexpr auto n_fields          = detail::deduceNFields(problem_def);
    constexpr auto scalar_inds       = std::array< size_t, 2 >{0, 2};
    constexpr auto vec_inds          = std::array< size_t, 2 >{1, 3};
    constexpr auto all_field_inds    = makeIotaArray< size_t, n_fields >();

    const auto system_manager   = AlgebraicSystemManager{comm, my_partition, problemdef_ctwrpr};
    auto       solution_manager = SolutionManager{my_partition, comm, n_fields};

    auto         solution      = system_manager.makeSolutionMultiVector();
    auto         solution_view = solution->getDataNonConst(0);
    const double Re            = 40.;
    const double lambda        = Re / 2. - std::sqrt(Re * Re / 4. - 4. * pi * pi);
    computeValuesAtNodes(
        [&](const SpaceTimePoint& p) {
            Eigen::Vector2d retval;
            retval[0] = 1.;
            retval[1] = pi;
            return retval;
        },
        my_partition,
        std::array{bot_boundary, top_boundary},
        system_manager.getRhsMap(),
        ConstexprValue< scalar_inds >{},
        solution_view);
    computeValuesAtNodes(
        [&](const SpaceTimePoint& p) {
            Eigen::Vector2d retval;
            // Kovasznay flow velocity field
            retval[0] = 1. - std::exp(lambda * p.space.x()) * std::cos(2. * pi * p.space.y());
            retval[1] = lambda * std::exp(lambda * p.space.x()) * std::sin(2 * pi * p.space.y()) / (2. * pi);
            return retval;
        },
        my_partition,
        std::views::single(domain_id),
        system_manager.getRhsMap(),
        ConstexprValue< vec_inds >{},
        solution_view);
    solution_manager.updateSolution(
        my_partition, *solution->getVector(0), system_manager.getRhsMap(), all_field_inds, problemdef_ctwrpr);
    solution_manager.communicateSharedValues();

    auto       exporter        = PvtuExporter{my_partition, solution_manager.getNodeMap()};
    const auto field_names     = std::array{"C1"sv, "Cpi"sv, "vel"sv};
    const auto field_comp_inds = std::array< std::span< const size_t >, 3 >{
        std::span{std::addressof(scalar_inds[0]), 1}, std::span{std::addressof(scalar_inds[1]), 1}, vec_inds};
    exporter.exportSolution("test_results_2D", comm, solution_manager, field_names, field_comp_inds);
}

void vtkExportTest3D()
{
    const MpiComm comm;

    const auto node_dist = [] {
        std::array< double, 11 > retval;
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -1. + i * 2. / (retval.size() - 1);
        return retval;
    }();

    Mesh mesh;
    if (comm.getRank() == 0)
    {
        constexpr auto mesh_order = 2;
        mesh                      = makeCubeMesh(node_dist);
        mesh.getPartitions()[0].initDualGraph();
        mesh.getPartitions()[0] = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    }
    const auto my_partition = distributeMesh(comm, mesh, {1, 2, 3, 4, 5, 6});

    constexpr auto problem_def       = std::array{Pair{d_id_t{0}, std::array{true, true, true}}};
    constexpr auto problemdef_ctwrpr = ConstexprValue< problem_def >{};
    constexpr auto n_fields          = detail::deduceNFields(problem_def);
    constexpr auto field_inds        = makeIotaArray< size_t, n_fields >();

    const auto system_manager   = AlgebraicSystemManager{comm, my_partition, problemdef_ctwrpr};
    auto       solution_manager = SolutionManager{my_partition, comm, n_fields};

    auto solution      = system_manager.makeSolutionMultiVector();
    auto solution_view = solution->getDataNonConst(0);
    computeValuesAtNodes(
        [&](const SpaceTimePoint& point) {
            Eigen::Vector3d retval;
            const auto&     p = point.space;
            const auto      r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
            retval[0]         = r;
            retval[1]         = p.y();
            retval[2]         = p.z();
            return retval;
        },
        my_partition,
        std::views::single(0),
        system_manager.getRhsMap(),
        ConstexprValue< field_inds >{},
        solution_view);
    solution_manager.updateSolution(
        my_partition, *solution->getVector(0), system_manager.getRhsMap(), field_inds, problemdef_ctwrpr);
    solution_manager.communicateSharedValues();

    auto       exporter   = PvtuExporter{my_partition, solution_manager.getNodeMap()};
    const auto field_name = "vec3D"sv;
    exporter.exportSolution(
        "test_results_3D", comm, solution_manager, std::views::single(field_name), std::views::single(field_inds));
}

int main(int argc, char* argv[])
{
    L3sterScopeGuard scope_guard{argc, argv};

    vtkExportTest2D();
    vtkExportTest3D();

    // TODO: Programatically check whether the data was exported correctly. For the time being, check manually...
}