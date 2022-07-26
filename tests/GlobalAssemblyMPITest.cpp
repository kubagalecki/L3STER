#include "l3ster/assembly/AlgebraicSystemManager.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/assembly/GatherGlobalValues.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "Amesos2.hpp"

#define CHECK_THROWS(X)                                                                                                \
    try                                                                                                                \
    {                                                                                                                  \
        X;                                                                                                             \
        return EXIT_FAILURE;                                                                                           \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {}

using namespace lstr;

// Solve 2D diffusion problem
int main(int argc, char* argv[])
{
    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    const MpiComm comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5., 6., 7., 8.};
    constexpr auto   mesh_order = 2;
    auto             mesh       = makeSquareMesh(node_dist);
    mesh.getPartitions()[0].initDualGraph();
    mesh.getPartitions()[0]    = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    constexpr d_id_t domain_id = 0, bot_boundary = 1, top_boundary = 2, left_boundary = 3, right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});
    const auto adiabatic_bound_view = my_partition.getBoundaryView(std::array{bot_boundary, top_boundary});
    const auto whole_bound_view =
        my_partition.getBoundaryView(std::array{top_boundary, bot_boundary, left_boundary, right_boundary});

    constexpr auto problem_def   = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true, true, true}}} >{};
    constexpr auto dirichlet_def = ConstexprValue< std::array{Pair{left_boundary, std::array{true, false, false}},
                                                              Pair{right_boundary, std::array{true, false, false}}} >{};

    // Problem assembly
    constexpr auto diffusion_kernel2d = [](const auto&, const auto&, const auto&) noexcept {
        using mat_t = Eigen::Matrix< val_t, 4, 3 >;
        std::pair< std::array< mat_t, 3 >, Eigen::Vector4d > retval;
        auto& [matrices, rhs] = retval;
        auto& [A0, A1, A2]    = matrices;
        constexpr double k    = 1.; // diffusivity

        A0       = mat_t::Zero();
        A1       = mat_t::Zero();
        A2       = mat_t::Zero();
        rhs      = Eigen::Vector4d::Zero();
        A0(1, 1) = -1.;
        A0(2, 2) = -1.;
        A1(0, 1) = k;
        A1(1, 0) = 1.;
        A1(3, 2) = 1.;
        A2(0, 2) = k;
        A2(2, 0) = 1;
        A2(3, 1) = -1;
        return retval;
    };
    static_assert(detail::Kernel_c< decltype(diffusion_kernel2d), 2, 0 >);
    constexpr auto neumann_bc_kernel =
        [](const auto&, const auto&, const auto&, const Eigen::Matrix< val_t, 2, 1 >& normal) noexcept {
            using mat_t = Eigen::Matrix< val_t, 1, 3 >;
            using vec_t = Eigen::Matrix< val_t, 1, 1 >;
            std::pair< std::array< mat_t, 3 >, vec_t > retval;
            auto& [matrices, rhs] = retval;
            auto& [A0, A1, A2]    = matrices;

            A0       = mat_t::Zero();
            A1       = mat_t::Zero();
            A2       = mat_t::Zero();
            rhs      = vec_t::Zero();
            A0(0, 1) = normal[0];
            A0(0, 2) = normal[1];
            return retval;
        };
    constexpr auto fieldval_getter = []< size_t N >(const std::array< n_id_t, N >& dofs) {
        return Eigen::Matrix< val_t, N, 0 >{};
    };

    constexpr auto BT       = BasisTypes::Lagrange;
    constexpr auto QT       = QuadratureTypes::GLeg;
    constexpr auto QO       = q_o_t{mesh_order * 2};
    constexpr auto dof_inds = std::array{size_t{0}, size_t{1}, size_t{2}};

    auto       problem               = AlgebraicSystemManager{comm, my_partition, problem_def, dirichlet_def};
    const auto assembleDomainProblem = [&] {
        problem.assembleDomainProblem< BT, QT, QO, dof_inds >(
            diffusion_kernel2d, my_partition, std::views::single(domain_id), fieldval_getter);
    };
    const auto assembleBoundaryProblem = [&] {
        problem.assembleBoundaryProblem< BT, QT, QO, dof_inds >(
            neumann_bc_kernel, adiabatic_bound_view, fieldval_getter);
    };

    problem.beginAssembly();
    CHECK_THROWS(problem.beginModify())
    CHECK_THROWS(problem.endModify())
    problem.endAssembly();
    CHECK_THROWS(problem.setToZero())
    CHECK_THROWS(assembleDomainProblem())
    CHECK_THROWS(assembleBoundaryProblem())
    problem.beginAssembly();
    problem.setToZero();
    assembleDomainProblem();
    assembleBoundaryProblem();
    problem.endAssembly();
    problem.endAssembly();

    // Dirichlet BCs
    constexpr auto dirichlet_bc_val_def = [node_dist](const SpaceTimePoint& st) {
        Eigen::Matrix< val_t, 1, 1 > retval;
        retval[0] = st.space.x() / node_dist.back();
        return retval;
    };
    auto dirichlet_vals = problem.makeCompatibleMultiVector(1u);
    computeValuesAtNodes< std::array{size_t{0}} >(dirichlet_bc_val_def,
                                                  my_partition,
                                                  std::array{left_boundary, right_boundary},
                                                  problem.getNodeToDofMap(),
                                                  *dirichlet_vals->getVectorNonConst(0));
    CHECK_THROWS(problem.applyDirichletBCs(*dirichlet_vals->getVector(0)))
    problem.beginModify();
    problem.beginModify();
    CHECK_THROWS(problem.beginAssembly())
    CHECK_THROWS(problem.endAssembly())
    problem.applyDirichletBCs(*dirichlet_vals->getVector(0));
    problem.endModify();
    problem.endModify();

    {
        auto fake_problem = AlgebraicSystemManager{comm, my_partition, problem_def};
        fake_problem.endAssembly();
        CHECK_THROWS(fake_problem.applyDirichletBCs(*dirichlet_vals->getVector(0)))
    }

    // Solve
    const auto glob_result = problem.makeCompatibleFEMultiVector(1u);
    glob_result->switchActiveMultiVector();
    Amesos2::KLU2< Tpetra::CrsMatrix<>, Tpetra::MultiVector<> > solver{
        problem.getMatrix(), glob_result, problem.getRhs()};
    solver.preOrdering().symbolicFactorization().numericFactorization().solve();

    // Check results
    glob_result->doOwnedToOwnedPlusShared(Tpetra::REPLACE);
    glob_result->switchActiveMultiVector();
    const auto     results_view         = glob_result->getVector(0)->getData();
    const auto&    dof_local_global_map = *glob_result->getMap();
    constexpr auto compute_error =
        [node_dist](const auto& vals, const auto& ders, const SpaceTimePoint& point) noexcept {
            Eigen::Matrix< val_t, 3, 1 > error;
            const auto                   T  = vals[0];
            const auto                   Tx = vals[1];
            const auto                   Ty = vals[2];
            error[0]                        = T - point.space.x() / node_dist.back();
            error[1]                        = Tx - 1. / node_dist.back();
            error[2]                        = Ty;
            return error;
        };
    constexpr auto compute_boundary_error =
        [compute_error](const auto& vals, const auto& ders, const SpaceTimePoint& point, const auto&) noexcept {
            return compute_error(vals, ders, point);
        };
    const auto fval_getter = [&]< size_t N >(const std::array< n_id_t, N >& nodes) {
        return gatherGlobalValues< std::array{size_t{0}, size_t{1}, size_t{2}} >(
            nodes, problem.getNodeToDofMap(), results_view, dof_local_global_map);
    };
    const auto error =
        computeNormL2< BT, QT, QO >(comm, compute_error, my_partition, std::views::single(domain_id), fval_getter);
    if (comm.getRank() == 0 and error.norm() > 1e-10)
    {
        std::stringstream error_msg;
        error_msg << "The error exceeded the allowed tolerance. The L2 error components were:\nvalue:\t\t\t" << error[0]
                  << "\nx derivative:\t" << error[1] << "\ny derivative:\t" << error[2] << '\n';
        std::cerr << error_msg.view();
        return EXIT_FAILURE;
    }
    const auto boundary_error =
        computeBoundaryNormL2< BT, QT, QO >(comm, compute_boundary_error, whole_bound_view, fval_getter);
    if (comm.getRank() == 0 and boundary_error.norm() > 1e-10)
    {
        std::cerr << "The error on the boundary exceeded the allowed tolerance. Since the error in the domain did not, "
                     "this is in all likelyhood an error in the computation of the norm on the boundary";
        return EXIT_FAILURE;
    }
}
