#define L3STER_ELEMENT_ORDERS 8
#include "l3ster/l3ster.hpp"

#include "DataPath.h"

int main(int argc, char* argv[])
{
    using namespace lstr;
    using namespace std::string_view_literals;

    L3sterScopeGuard scope_guard{argc, argv};
    const MpiComm    comm{MPI_COMM_WORLD};

    constexpr d_id_t      domain_id           = 0;
    static constexpr auto boundary_ids        = util::makeIotaArray< d_id_t, 6 >(1);
    constexpr auto        problem_def         = std::array{Pair{domain_id, std::array{true, true, true, true}}};
    constexpr auto        dirichlet_def       = std::invoke([] {
        std::array< Pair< d_id_t, std::array< bool, 4 > >, boundary_ids.size() > retval{};
        std::ranges::transform(boundary_ids, retval.begin(), [](auto d) {
            return Pair{d, std::array{true, false, false, false}};
        });
        return retval;
    });
    constexpr auto        probdef_ctwrpr      = ConstexprValue< problem_def >{};
    constexpr auto        dirichletdef_ctwrpr = ConstexprValue< dirichlet_def >{};

    constexpr auto node_dist    = std::invoke([] {
        constexpr size_t                   edge_divs = 2;
        constexpr auto                     dx        = 1. / static_cast< val_t >(edge_divs);
        std::array< val_t, edge_divs + 1 > retval{};
        for (double x = 0; auto& r : retval)
        {
            r = x;
            x += dx;
        }
        return retval;
    });
    constexpr auto mesh_order   = L3STER_ELEMENT_ORDERS;
    const auto     my_partition = generateAndDistributeMesh< mesh_order >(
        comm,
        [&] { return makeCubeMesh(node_dist); },
        std::vector< d_id_t >(boundary_ids.begin(), boundary_ids.end()),
        {},
        probdef_ctwrpr);
    const auto boundary_view = my_partition.getBoundaryView(boundary_ids);

    constexpr auto n_fields    = detail::deduceNFields(problem_def);
    constexpr auto field_inds  = util::makeIotaArray< size_t, n_fields >();
    constexpr auto T_inds      = std::array< size_t, 1 >{0};
    constexpr auto T_grad_inds = std::array< size_t, 3 >{1, 2, 3};
    const auto field_inds_view = std::array{std::span< const size_t >{T_inds}, std::span< const size_t >{T_grad_inds}};
    constexpr auto field_names = std::array{"T"sv, "gradT"sv};
    constexpr auto dof_inds    = field_inds;

    constexpr auto diffusion_kernel3d =
        []< typename T >(const auto&, const std::array< T, 3 >&, const SpaceTimePoint&) noexcept {
            using mat_t = Eigen::Matrix< val_t, 7, 4 >;
            using vec_t = Eigen::Vector< double, 7 >;
            std::pair< std::array< mat_t, 4 >, vec_t > retval;
            auto& [matrices, rhs]  = retval;
            auto& [A0, Ax, Ay, Az] = matrices;

            constexpr double k = 1.; // diffusivity
            constexpr double s = 1.; // source

            A0  = mat_t::Zero();
            Ax  = mat_t::Zero();
            Ay  = mat_t::Zero();
            Az  = mat_t::Zero();
            rhs = vec_t::Zero();

            // -k * div q = s
            Ax(0, 1) = -k;
            Ay(0, 2) = -k;
            Az(0, 3) = -k;
            rhs[0]   = s;

            // grad T = q
            A0(1, 1) = -1.;
            Ax(1, 0) = 1.;
            A0(2, 2) = -1.;
            Ay(2, 0) = 1.;
            A0(3, 3) = -1.;
            Az(3, 0) = 1.;

            // rot q = 0
            Ay(4, 3) = 1.;
            Az(4, 2) = -1.;
            Ax(5, 3) = -1.;
            Az(5, 1) = 1.;
            Ax(6, 2) = 1.;
            Ay(6, 1) = -1.;

            return retval;
        };
    constexpr auto error_kernel =
        []< typename DerT >(const auto& vals, const std::array< DerT, 3 >& ders, const SpaceTimePoint& point) noexcept {
            Eigen::Matrix< val_t, 4, 1 > error;
            const auto [T, qx, qy, qz]           = vals;
            const auto& [x_ders, y_ders, z_ders] = ders;

            const auto& q_xx = x_ders[1];
            const auto& q_yy = y_ders[2];
            const auto& q_zz = z_ders[3];
            const auto& T_x  = x_ders[0];
            const auto& T_y  = y_ders[0];
            const auto& T_z  = z_ders[0];

            constexpr double k = 1.;
            constexpr double s = 1.;

            error[0] = k * (q_xx + q_yy + q_zz) + s;
            error[1] = T_x - qx;
            error[2] = T_y - qy;
            error[3] = T_z - qz;
            return error;
        };
    constexpr auto dirichlet_bc_val_def = [](const auto&, const auto&, const SpaceTimePoint& p) {
        Eigen::Vector< val_t, 1 > retval;
        retval[0] = 0.;
        return retval;
    };

    auto alg_system = makeAlgebraicSystem(comm, my_partition, element_boundary, probdef_ctwrpr, dirichletdef_ctwrpr);
    alg_system->beginAssembly();
    alg_system->assembleDomainProblem(diffusion_kernel3d, my_partition, std::views::single(domain_id));
    alg_system->endAssembly(my_partition);

    alg_system->describe(comm);

    alg_system->setDirichletBCValues(dirichlet_bc_val_def, my_partition, boundary_ids, ConstexprValue< T_inds >{});
    alg_system->applyDirichletBCs();

    solvers::CG solver{
        1e-6, 10'000, static_cast< Belos::MsgType >(Belos::Warnings + Belos::FinalSummary + Belos::TimingDetails)};
    auto solution = alg_system->makeSolutionVector();
    alg_system->solve(solver, solution);

    L3STER_PROFILE_REGION_BEGIN("Solution management");
    auto solution_manager = SolutionManager{my_partition, n_fields};
    alg_system->updateSolution(my_partition, solution, dof_inds, solution_manager, field_inds);
    L3STER_PROFILE_REGION_END("Solution management");

    L3STER_PROFILE_REGION_BEGIN("Compute solution error");
    const auto fval_getter = solution_manager.makeFieldValueGetter(field_inds);
    const auto error = computeNormL2(comm, error_kernel, my_partition, std::views::single(domain_id), fval_getter);
    L3STER_PROFILE_REGION_END("Compute solution error");

    if (comm.getRank() == 0)
    {
        std::cout << "The L2 error components are:";
        for (int i = 0; i < error.size(); ++i)
            std::cout << "\n\t" << error[i];
        std::cout << std::endl;
    }

    L3STER_PROFILE_REGION_BEGIN("Export results to VTK");
    auto exporter = PvtuExporter{my_partition};
    exporter.exportSolution("Cube_Diffusion.pvtu", comm, solution_manager, field_names, field_inds_view);
    L3STER_PROFILE_REGION_END("Export results to VTK");
}