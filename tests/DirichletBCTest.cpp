#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/assembly/ScatterLocalSystem.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Amesos2.hpp"
#include "Tpetra_MultiVector.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;
    L3sterScopeGuard scope_guard{argc, argv};

    const MpiComm comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5.};
    constexpr auto   boundary     = 3;
    const auto       mesh         = makeCubeMesh(node_dist);
    auto             my_partition = distributeMesh(comm, mesh, {boundary});

    constexpr d_id_t domain_id      = 0;
    constexpr auto   problem_def    = ConstexprValue< std::array{Pair{d_id_t{domain_id}, std::array{true}}} >{};
    const auto       dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto       global_dof_map = NodeToGlobalDofMap{my_partition, dof_intervals};
    const auto       sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);

    const auto bv = my_partition.getBoundaryView(boundary);

    constexpr auto dirichlet_def = ConstexprValue< std::array{Pair{d_id_t{boundary}, std::array{true}}} >{};
    const auto& [owned_bcdofs, shared_bcdofs] =
        detail::getDirichletDofs(my_partition, sparsity_graph, global_dof_map, problem_def, dirichlet_def);
    const auto dirichlet_bc = DirichletBCAlgebraic{sparsity_graph, owned_bcdofs, shared_bcdofs};

    auto                   matrix = makeTeuchosRCP< tpetra_fecrsmatrix_t >(sparsity_graph);
    tpetra_femultivector_t input_vectors{sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 2};
    tpetra_multivector_t   result_vectors{sparsity_graph->getRowMap(), 2};
    matrix->beginAssembly();
    input_vectors.beginAssembly();

    const auto dof_map = NodeToLocalDofMap{
        my_partition, global_dof_map, *matrix->getRowMap(), *matrix->getColMap(), *input_vectors.getMap()};

    {
        auto rhs      = input_vectors.getVectorNonConst(0)->getDataNonConst();
        auto rhs_view = std::span{rhs};
        my_partition.visit(
            [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                if constexpr (T == ElementTypes::Hex and O == 1)
                {
                    constexpr int n_dofs = Element< T, O >::n_nodes;
                    using local_mat_t    = EigenRowMajorMatrix< val_t, n_dofs, n_dofs >;
                    using local_vec_t    = Eigen::Matrix< val_t, n_dofs, 1 >;
                    std::pair< local_mat_t, local_vec_t > local_system;
                    auto& [local_mat, local_vec] = local_system;
                    local_mat                    = local_mat_t::Random();
                    local_mat                    = local_mat.template selfadjointView< Eigen::Lower >();
                    local_vec                    = local_vec_t::Random();

                    const auto [row_dofs, col_dofs, rhs_dofs] =
                        detail::getUnsortedElementDofs< std::array{size_t{0}} >(element, dof_map);
                    detail::scatterLocalSystem(local_mat, local_vec, *matrix, rhs_view, row_dofs, col_dofs, rhs_dofs);
                }
            },
            std::views::single(domain_id));
    }
    matrix->endAssembly();
    input_vectors.endAssembly();

    input_vectors.beginModify();
    {
        auto bc_vals = input_vectors.getVectorNonConst(1);
        bc_vals->randomize();
    }
    input_vectors.endModify();

    matrix->beginModify();
    input_vectors.beginModify();
    {
        auto rhs     = input_vectors.getVectorNonConst(0);
        auto bc_vals = input_vectors.getVector(1);
        dirichlet_bc.apply(*bc_vals, *matrix, *rhs);
    }
    matrix->endModify();
    input_vectors.endModify();

    const auto rhs_vector     = input_vectors.getVector(0);
    const auto bc_val_vector  = input_vectors.getVector(1);
    auto       result_vector1 = result_vectors.getVectorNonConst(0);
    auto       result_vector2 = result_vectors.getVectorNonConst(1);

    constexpr auto approx = [](double a, double b) {
        return std::fabs(a - b) < 1e-10;
    };

    matrix->apply(*bc_val_vector, *result_vector1, Teuchos::NO_TRANS);
    matrix->apply(*bc_val_vector, *result_vector2, Teuchos::TRANS);
    const auto data_notransp =
        Kokkos::subview(result_vector1->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    const auto data_transp =
        Kokkos::subview(result_vector2->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    for (int i = 0; i < data_notransp.extent_int(0); ++i)
        if (not approx(data_notransp[i], data_transp[i]))
        {
            std::cerr << "Application of Dirichlet BC broke the symmetry of the matrix\n";
            comm.abort();
        }

    Amesos2::KLU2< tpetra_crsmatrix_t, tpetra_multivector_t > solver{matrix, result_vector1, rhs_vector};

    if (not solver.matrixShapeOK())
    {
        std::cerr << "Bad matrix shape\n";
        comm.abort();
    }

    solver.preOrdering().symbolicFactorization().numericFactorization().solve();

    const auto solution_vals =
        Kokkos::subview(result_vector1->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    const auto bc_vals = Kokkos::subview(bc_val_vector->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    for (auto global_bcdof : owned_bcdofs)
    {
        const auto local_bcdof = matrix->getRowMap()->getLocalElement(global_bcdof);
        if (not approx(solution_vals[local_bcdof], bc_vals[local_bcdof]))
        {
            std::cerr << "Result of solve does not satisfy the Dirichlet BC\n";
            comm.abort();
        }
    }
}
