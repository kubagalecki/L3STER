#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/assembly/ContributeLocalSystem.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"

#include "Amesos2.hpp"
#include "Tpetra_MultiVector.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;

    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    const MpiComm comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5.};
    constexpr auto   boundary     = 3;
    const auto       mesh         = makeCubeMesh(node_dist);
    auto             my_partition = distributeMesh(comm, mesh, {boundary});

    constexpr auto problem_def    = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true}}} >{};
    const auto     dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto     map            = NodeToDofMap{my_partition, dof_intervals};
    const auto     sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);

    const auto bv = my_partition.getBoundaryView(boundary);

    constexpr auto dirichlet_def = ConstexprValue< std::array{Pair{d_id_t{boundary}, std::array{true}}} >{};
    const auto& [owned_bcdofs, shared_bcdofs] =
        detail::getDirichletDofs(my_partition, sparsity_graph, map, problem_def, dirichlet_def);
    const auto dirichlet_bc = DirichletBCAlgebraic{sparsity_graph, owned_bcdofs, shared_bcdofs};

    auto                    matrix = makeTeuchosRCP< Tpetra::FECrsMatrix<> >(sparsity_graph);
    Tpetra::FEMultiVector<> input_vectors{sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 2};
    Tpetra::MultiVector<>   result_vectors{sparsity_graph->getRowMap(), 2};
    matrix->beginAssembly();
    input_vectors.beginAssembly();
    {
        auto rhs = input_vectors.getVectorNonConst(0);
        my_partition.visit(
            [&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
                if constexpr (T == ElementTypes::Hex and O == 1)
                {
                    constexpr int n_dofs = Element< T, O >::n_nodes;
                    using local_mat_t    = Eigen::Matrix< val_t, n_dofs, n_dofs, Eigen::RowMajor >;
                    using local_vec_t    = Eigen::Matrix< val_t, n_dofs, 1 >;
                    std::pair< local_mat_t, local_vec_t > local_system;
                    auto& [local_mat, local_vec] = local_system;
                    local_mat                    = local_mat_t::Random();
                    local_mat                    = local_mat.template selfadjointView< Eigen::Lower >();
                    local_vec                    = local_vec_t::Random();
                    contributeLocalSystem< std::array{ptrdiff_t{0}} >(local_system, element, map, *matrix, *rhs);
                }
            },
            std::views::single(0));
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
    const auto data_notransp = result_vector1->getData();
    const auto data_transp   = result_vector2->getData();
    for (auto i : std::views::iota(0, data_notransp.size()))
        if (not approx(data_notransp[i], data_transp[i]))
        {
            std::cerr << "Application of Dirichlet BC broke the symmetry of the matrix\n";
            return 1;
        }

    Amesos2::KLU2< Tpetra::CrsMatrix<>, Tpetra::MultiVector<> > solver{matrix, result_vector1, rhs_vector};

    if (not solver.matrixShapeOK())
    {
        std::cerr << "Bad matrix shape\n";
        return 1;
    }

    solver.preOrdering().symbolicFactorization().numericFactorization().solve();

    const auto solution_vals = result_vector1->getData();
    const auto bc_vals       = bc_val_vector->getData();
    for (auto global_bcdof : owned_bcdofs)
    {
        const auto local_bcdof = matrix->getRowMap()->getLocalElement(global_bcdof);
        if (not approx(solution_vals[local_bcdof], bc_vals[local_bcdof]))
        {
            std::cerr << "Result of solve does not satisfy the Dirichlet BC\n";
            return 1;
        }
    }
}
