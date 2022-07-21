#include "l3ster/assembly/AssembleGlobalSystem.hpp"
#include "l3ster/assembly/ComputeValuesAtNodes.hpp"
#include "l3ster/assembly/GatherGlobalValues.hpp"
#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/ConvertMeshToOrder.hpp"
#include "l3ster/mesh/primitives/SquareMesh.hpp"
#include "l3ster/post/NormL2.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "Amesos2.hpp"

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
    mesh.getPartitions()[0]         = convertMeshToOrder< mesh_order >(mesh.getPartitions()[0]);
    constexpr d_id_t domain_id      = 0;
    constexpr d_id_t bot_boundary   = 1;
    constexpr d_id_t top_boundary   = 2;
    constexpr d_id_t left_boundary  = 3;
    constexpr d_id_t right_boundary = 4;
    const auto my_partition = distributeMesh(comm, mesh, {bot_boundary, top_boundary, left_boundary, right_boundary});
    const auto adiabatic_bound_view = my_partition.getBoundaryView(std::array{bot_boundary, top_boundary});
    const auto whole_bound_view =
        my_partition.getBoundaryView(std::array{top_boundary, bot_boundary, left_boundary, right_boundary});

    constexpr auto problem_def    = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true, true, true}}} >{};
    const auto     dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto     map            = NodeToDofMap{my_partition, dof_intervals};
    const auto     sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);

    constexpr auto dirichlet_def = ConstexprValue< std::array{Pair{left_boundary, std::array{true, false, false}},
                                                              Pair{right_boundary, std::array{true, false, false}}} >{};
    const auto& [owned_bcdofs, shared_bcdofs] =
        detail::getDirichletDofs(my_partition, sparsity_graph, map, problem_def, dirichlet_def);
    const auto     dirichlet_bc         = DirichletBCAlgebraic{sparsity_graph, owned_bcdofs, shared_bcdofs};
    constexpr auto dirichlet_bc_val_def = [node_dist](const SpaceTimePoint& st) {
        Eigen::Matrix< val_t, 1, 1 > retval;
        retval[0] = st.space.x() / node_dist.back();
        return retval;
    };

    const auto glob_mat = makeTeuchosRCP< Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t > >(sparsity_graph);
    const auto glob_rhs = makeTeuchosRCP< Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t > >(
        sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1u);
    const auto glob_result = makeTeuchosRCP< Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t > >(
        sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1u);
    glob_result->beginAssembly();
    glob_result->endAssembly();

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

    // Problem assembly
    glob_mat->beginAssembly();
    glob_rhs->beginAssembly();
    {
        const auto rhs_asm_vec = glob_rhs->getVectorNonConst(0);
        rhs_asm_vec->modify_host();
        const auto rhs_data = rhs_asm_vec->getDataNonConst();
        assembleGlobalSystem< BT, QT, QO, dof_inds >(diffusion_kernel2d,
                                                     my_partition,
                                                     std::views::single(domain_id),
                                                     map,
                                                     fieldval_getter,
                                                     *glob_mat,
                                                     rhs_data,
                                                     *glob_rhs->getMap());
        assembleGlobalBoundarySystem< BT, QT, QO, dof_inds >(
            neumann_bc_kernel, adiabatic_bound_view, map, fieldval_getter, *glob_mat, rhs_data, *glob_rhs->getMap());
        rhs_asm_vec->sync_device();
    }
    glob_mat->endAssembly();
    glob_rhs->endAssembly();

    // Dirichlet BCs
    Tpetra::Vector dirichlet_vals{glob_rhs->getMap()};
    computeValuesAtNodes< std::array{size_t{0}} >(
        dirichlet_bc_val_def, my_partition, std::array{left_boundary, right_boundary}, map, dirichlet_vals);

    glob_mat->beginModify();
    glob_rhs->beginModify();
    dirichlet_bc.apply(dirichlet_vals, *glob_mat, *glob_rhs->getVectorNonConst(0));
    glob_mat->endModify();
    glob_rhs->endModify();

    // Solve
    Amesos2::KLU2< Tpetra::CrsMatrix<>, Tpetra::MultiVector<> > solver{glob_mat, glob_result, glob_rhs};
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
            nodes, map, results_view, dof_local_global_map);
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
