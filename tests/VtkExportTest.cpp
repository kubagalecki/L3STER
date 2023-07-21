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
using namespace std::numbers;
using namespace std::string_view_literals;

void vtkExportTest2D()
{
    const MpiComm comm{MPI_COMM_WORLD};

    constexpr auto   node_distx = std::invoke([] {
        auto retval = std::array< double, 11 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + static_cast< double >(i) * 2.5 / (retval.size() - 1);
        return retval;
    });
    constexpr auto   node_disty = std::invoke([] {
        auto retval = std::array< double, 15 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -.5 + static_cast< double >(i) * 2. / (retval.size() - 1);
        return retval;
    });
    constexpr auto   mesh_order = 2;
    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    const auto       my_partition =
        generateAndDistributeMesh< mesh_order >(comm,
                                                [&] { return mesh::makeSquareMesh(node_distx, node_disty); },
                                                {bot_boundary, top_boundary, left_boundary, right_boundary});

    constexpr auto problem_def       = ProblemDef{defineDomain< 4 >(domain_id, 1, 3),
                                            defineDomain< 4 >(bot_boundary, 0, 2),
                                            defineDomain< 4 >(top_boundary, 0, 2)};
    constexpr auto problemdef_ctwrpr = util::ConstexprValue< problem_def >{};
    constexpr auto scalar_inds       = std::array< size_t, 2 >{0, 2};
    constexpr auto vec_inds          = std::array< size_t, 2 >{1, 3};
    constexpr auto all_field_inds    = util::makeIotaArray< size_t, problem_def.n_fields >();

    const auto system_manager   = makeAlgebraicSystem(comm, my_partition, no_condensation_tag, problemdef_ctwrpr);
    auto       solution_manager = SolutionManager{my_partition, problem_def.n_fields};

    auto solution = system_manager->makeSolutionVector();
    {
        auto         solution_view = solution->get1dViewNonConst();
        const double Re            = 40.;
        const double lambda        = Re / 2. - std::sqrt(Re * Re / 4. - 4. * pi * pi);
        const auto   bot_top_vals  = std::array< val_t, 2 >{1., pi};
        computeValuesAtNodes(my_partition,
                             std::array{bot_boundary, top_boundary},
                             system_manager->getDofMap(),
                             util::ConstexprValue< scalar_inds >{},
                             bot_top_vals,
                             solution_view);
        computeValuesAtNodes(
            [&]([[maybe_unused]] const auto& vals, [[maybe_unused]] const auto& ders, const SpaceTimePoint& p) {
                Eigen::Vector2d retval;
                // Kovasznay flow velocity field
                retval[0] = 1. - std::exp(lambda * p.space.x()) * std::cos(2. * pi * p.space.y());
                retval[1] = lambda * std::exp(lambda * p.space.x()) * std::sin(2 * pi * p.space.y()) / (2. * pi);
                return retval;
            },
            my_partition,
            std::views::single(domain_id),
            system_manager->getDofMap(),
            util::ConstexprValue< vec_inds >{},
            empty_field_val_getter,
            solution_view);
    }
    solution->switchActiveMultiVector();
    solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    solution->switchActiveMultiVector();
    system_manager->updateSolution(my_partition, solution, all_field_inds, solution_manager, all_field_inds);

    auto       exporter        = PvtuExporter{my_partition};
    const auto field_names     = std::array{"C1"sv, "Cpi"sv, "vel"sv};
    const auto field_comp_inds = std::array< std::span< const size_t >, 3 >{
        std::span{std::addressof(scalar_inds[0]), 1}, std::span{std::addressof(scalar_inds[1]), 1}, vec_inds};
    if (comm.getRank() == 0)
    {
        [[maybe_unused]] auto _ = system("mkdir -p 2D");
    }
    comm.barrier();
    exporter.exportSolution("2D/results", comm, solution_manager, field_names, field_comp_inds);
}

void vtkExportTest3D()
{
    const MpiComm comm{MPI_COMM_WORLD};

    const auto node_dist = [] {
        auto retval = std::array< double, 7 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -1. + static_cast< double >(i) * 2. / (retval.size() - 1);
        return retval;
    }();
    constexpr auto mesh_order   = 2;
    const auto     my_partition = generateAndDistributeMesh< mesh_order >(
        comm, [&] { return mesh::makeCubeMesh(node_dist); }, {1, 2, 3, 4, 5, 6});
    const auto boundary = my_partition.getBoundaryView(util::makeIotaArray< d_id_t, 6 >(1));

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

    const auto system_manager   = makeAlgebraicSystem(comm, my_partition, no_condensation_tag, problemdef_ctwrpr);
    auto       solution_manager = SolutionManager{my_partition, n_fields};

    auto solution = system_manager->makeSolutionVector();
    {
        auto solution_view = solution->get1dViewNonConst();
        computeValuesAtNodes(
            [&]([[maybe_unused]] const auto& vals, [[maybe_unused]] const auto& ders, const SpaceTimePoint& point) {
                Eigen::Vector3d retval;
                const auto&     p = point.space;
                const auto      r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
                retval[0]         = r;
                retval[1]         = p.y();
                retval[2]         = p.z();
                return retval;
            },
            my_partition,
            std::views::single(domain_id),
            system_manager->getDofMap(),
            util::ConstexprValue< domain_field_inds >{},
            empty_field_val_getter,
            solution_view);
        computeValuesAtBoundaryNodes([&](const auto&,
                                         const std::array< std::array< val_t, 0 >, 3 >&,
                                         const auto&,
                                         const Eigen::Vector3d& normal) -> Eigen::Vector3d { return normal; },
                                     boundary,
                                     system_manager->getDofMap(),
                                     util::ConstexprValue< boundary_field_inds >{},
                                     empty_field_val_getter,
                                     solution_view);

        solution->switchActiveMultiVector();
        solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
        solution->switchActiveMultiVector();
    }
    constexpr auto field_inds = util::makeIotaArray< size_t, n_fields >();
    system_manager->updateSolution(my_partition, solution, field_inds, solution_manager, field_inds);

    auto       exporter    = PvtuExporter{my_partition};
    const auto field_names = std::array{"vec3D"sv, "normal"sv};
    CHECK_THROWS(exporter.exportSolution("path/to/nonexistent/directory/test_results_3D",
                                         comm,
                                         solution_manager,
                                         field_names,
                                         std::array{domain_field_inds, boundary_field_inds}));
    if (comm.getRank() == 0)
    {
        [[maybe_unused]] auto _ = system("mkdir -p 3D");
    }
    comm.barrier();
    exporter.exportSolution("3D/results.nonsense_extension",
                            comm,
                            solution_manager,
                            field_names,
                            std::array{domain_field_inds, boundary_field_inds});
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};

    vtkExportTest2D();
    vtkExportTest3D();

    // TODO: Programatically check whether the data was exported correctly. For the time being, check manually...
}