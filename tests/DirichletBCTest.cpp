#include "l3ster/bcs/DirichletBC.hpp"
#include "l3ster/algsys/ScatterLocalSystem.hpp"
#include "l3ster/algsys/SparsityGraph.hpp"
#include "l3ster/bcs/GetDirichletDofs.hpp"
#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/mesh/primitives/CubeMesh.hpp"
#include "l3ster/solve/Amesos2Solvers.hpp"
#include "l3ster/util/ScopeGuards.hpp"

#include "Common.hpp"

using namespace lstr;
using namespace lstr::bcs;
using namespace lstr::dofs;
using namespace lstr::algsys;
using namespace lstr::mesh;

template < CondensationPolicy CP >
void test()
{
    const MpiComm comm{MPI_COMM_WORLD};

    constexpr auto   boundary      = 3;
    constexpr d_id_t domain_id     = 0;
    const auto       problem_def   = ProblemDefinition< 1 >{{domain_id}};
    auto             dirichlet_def = DirichletBCDefinition< 1 >{};
    dirichlet_def.defineDirichletBoundary({boundary}, {0});

    constexpr auto node_dist    = std::array{0., 1., 2., 3., 4., 5.};
    auto           my_partition = comm::distributeMesh(comm, [&] { return makeCubeMesh(node_dist); });

    constexpr auto cp_tag              = CondensationPolicyTag< CP >{};
    const auto     global_node_dof_map = NodeToGlobalDofMap{comm, *my_partition, problem_def, {}, cp_tag};
    const auto     sparsity_graph = makeSparsityGraph(comm, *my_partition, global_node_dof_map, problem_def, cp_tag);

    const auto [owned_bcdofs, shared_bcdofs] =
        getDirichletDofs(*my_partition, sparsity_graph, global_node_dof_map, dirichlet_def);
    const auto dirichlet_bc = DirichletBCAlgebraic{sparsity_graph, owned_bcdofs, shared_bcdofs};

    auto matrix         = util::makeTeuchosRCP< tpetra_fecrsmatrix_t >(sparsity_graph);
    auto input_vectors  = tpetra_femultivector_t{sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 2};
    auto result_vectors = tpetra_multivector_t{sparsity_graph->getRowMap(), 2};

    matrix->beginAssembly();
    input_vectors.beginAssembly();

    const auto dof_map =
        NodeToLocalDofMap{global_node_dof_map, *matrix->getRowMap(), *matrix->getColMap(), *input_vectors.getMap()};

    {
        auto rhs      = input_vectors.getVectorNonConst(0)->getDataNonConst();
        auto rhs_view = std::span{rhs};
        my_partition->visit(
            [&]< ElementType T, el_o_t O >(const Element< T, O >& element) {
                if constexpr (T == ElementType::Hex and O == 1)
                {
                    constexpr int n_dofs    = getNumPrimaryNodes< CP, T, O >() * /* dofs per node */ 1;
                    auto          local_mat = util::eigen::RowMajorSquareMatrix< val_t, n_dofs >{};
                    auto          local_vec = Eigen::Vector< val_t, n_dofs >{};
                    local_mat.setRandom();
                    local_mat = local_mat.template selfadjointView< Eigen::Lower >();
                    local_vec.setRandom();

                    constexpr auto dof_inds_wrpr = util::ConstexprValue< std::array{size_t{0}} >{};
                    const auto [row_dofs, col_dofs, rhs_dofs] =
                        getDofsFromNodes(getPrimaryNodesArray< CP >(element), dof_map, dof_inds_wrpr);
                    scatterLocalSystem(
                        local_mat, local_vec, *matrix, std::array{rhs_view}, row_dofs, col_dofs, rhs_dofs);
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
        REQUIRE(approx(data_notransp[i], data_transp[i]));

    auto solver = solvers::KLU2{};
    solver.solve(matrix, rhs_vector, result_vector1);

    const auto solution_vals =
        Kokkos::subview(result_vector1->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    const auto bc_vals = Kokkos::subview(bc_val_vector->getLocalViewHost(Tpetra::Access::ReadOnly), Kokkos::ALL, 0);
    for (auto global_bcdof : owned_bcdofs)
    {
        const auto local_bcdof = matrix->getRowMap()->getLocalElement(global_bcdof);
        REQUIRE(approx(solution_vals[local_bcdof], bc_vals[local_bcdof]));
    }
}

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None >();
    test< CondensationPolicy::ElementBoundary >();
}
