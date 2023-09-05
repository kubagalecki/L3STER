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
using namespace std::numbers;
using namespace std::string_view_literals;

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};

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
    auto       solution_manager = SolutionManager{*my_partition, problem_def.n_fields};

    auto solution = system_manager->makeSolutionVector();
    {
        auto         solution_view = solution->get1dViewNonConst();
        const double Re            = 40.;
        const double lambda        = Re / 2. - std::sqrt(Re * Re / 4. - 4. * pi * pi);
        const auto   bot_top_vals  = std::array< val_t, 2 >{1., pi};
        computeValuesAtNodes(*my_partition,
                             std::array{bot_boundary, top_boundary},
                             system_manager->getDofMap(),
                             scalar_inds,
                             std::span{bot_top_vals},
                             solution_view);
        constexpr auto params = KernelParams{.dimension = 2, .n_equations = 2};
        const auto     kernel = wrapResidualDomainKernel< params >([&](const auto& in, auto& out) {
            // Kovasznay flow velocity field
            const auto& p = in.point.space;
            out[0]        = 1. - std::exp(lambda * p.x()) * std::cos(2. * pi * p.y());
            out[1]        = lambda * std::exp(lambda * p.x()) * std::sin(2 * pi * p.y()) / (2. * pi);
        });
        computeValuesAtNodes(kernel,
                             *my_partition,
                             std::views::single(domain_id),
                             system_manager->getDofMap(),
                             vec_inds,
                             empty_field_val_getter,
                             solution_view);
    }
    solution->switchActiveMultiVector();
    solution->doOwnedToOwnedPlusShared(Tpetra::CombineMode::REPLACE);
    solution->switchActiveMultiVector();
    system_manager->updateSolution(solution, all_field_inds, solution_manager, all_field_inds);

    auto       exporter        = PvtuExporter{*my_partition};
    const auto field_names     = std::array{"C1"sv, "Cpi"sv, "vel"sv};
    const auto field_comp_inds = std::array< std::span< const size_t >, 3 >{
        std::span{std::addressof(scalar_inds[0]), 1}, std::span{std::addressof(scalar_inds[1]), 1}, vec_inds};
    comm.barrier();
    exporter.exportSolution("results", comm, solution_manager, field_names, field_comp_inds);

    // TODO: Programatically check whether the data was exported correctly. For the time being, check manually...
}