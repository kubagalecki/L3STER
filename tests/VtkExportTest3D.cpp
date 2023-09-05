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

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};

    const MpiComm comm{MPI_COMM_WORLD};

    const auto node_dist = [] {
        auto retval = std::array< double, 7 >{};
        for (size_t i = 0; i < retval.size(); ++i)
            retval[i] = -1. + static_cast< double >(i) * 2. / (retval.size() - 1);
        return retval;
    }();
    constexpr auto mesh_order = 2;
    const auto     my_partition =
        generateAndDistributeMesh< mesh_order >(comm, [&] { return makeCubeMesh(node_dist); }, {1, 2, 3, 4, 5, 6});
    const auto boundary = BoundaryView{*my_partition, util::makeIotaArray< d_id_t, 6 >(1)};

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
    auto       solution_manager = SolutionManager{*my_partition, n_fields};

    auto solution = system_manager->makeSolutionVector();
    {
        auto           solution_view = solution->get1dViewNonConst();
        constexpr auto ker_params    = KernelParams{.dimension = 3, .n_equations = 3};

        const auto dom_kernel = wrapResidualDomainKernel< ker_params >([&](const auto& in, auto& out) {
            const auto& p = in.point.space;
            const auto  r = std::sqrt(p.x() * p.x() + p.y() * p.y() + p.z() * p.z());
            out[0]        = r;
            out[1]        = p.y();
            out[2]        = p.z();
        });
        computeValuesAtNodes(dom_kernel,
                             *my_partition,
                             std::views::single(domain_id),
                             system_manager->getDofMap(),
                             domain_field_inds,
                             empty_field_val_getter,
                             solution_view);

        const auto bnd_kernel =
            wrapResidualBoundaryKernel< ker_params >([&](const auto& in, auto& out) { out = in.normal; });
        computeValuesAtBoundaryNodes(bnd_kernel,
                                     boundary,
                                     system_manager->getDofMap(),
                                     boundary_field_inds,
                                     empty_field_val_getter,
                                     solution_view);

        solution->switchActiveMultiVector();
        solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
        solution->switchActiveMultiVector();
    }
    constexpr auto field_inds = util::makeIotaArray< size_t, n_fields >();
    system_manager->updateSolution(solution, field_inds, solution_manager, field_inds);

    auto       exporter    = PvtuExporter{*my_partition};
    const auto field_names = std::array{"vec3D"sv, "normal"sv};
    CHECK_THROWS(exporter.exportSolution("path/to/nonexistent/directory/test_results_3D",
                                         comm,
                                         solution_manager,
                                         field_names,
                                         std::array{domain_field_inds, boundary_field_inds}));
    comm.barrier();
    exporter.exportSolution("results.nonsense_extension",
                            comm,
                            solution_manager,
                            field_names,
                            std::array{domain_field_inds, boundary_field_inds});

    // TODO: Programatically check whether the data was exported correctly. For the time being, check manually...
}