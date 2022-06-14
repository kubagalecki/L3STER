#include "l3ster/comm/DistributeMesh.hpp"
#include "l3ster/global_assembly/ContributeLocalSystem.hpp"
#include "l3ster/mesh/primitives/LineMesh.hpp"
#include "l3ster/util/GlobalResource.hpp"

int main(int argc, char* argv[])
{
    using namespace lstr;

    GlobalResource< MpiScopeGuard >::initialize(argc, argv);
    const MpiComm comm;

    const std::array node_dist{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10.};
    const auto       mesh         = makeLineMesh(node_dist);
    auto             my_partition = distributeMesh(comm, mesh, {});

    constexpr auto problem_def    = ConstexprValue< std::array{Pair{d_id_t{0}, std::array{true}}} >{};
    const auto     dof_intervals  = computeDofIntervals(my_partition, problem_def, comm);
    const auto     map            = NodeToDofMap{my_partition, dof_intervals};
    const auto     sparsity_graph = detail::makeSparsityGraph(my_partition, problem_def, dof_intervals, comm);

    const auto glob_mat = Teuchos::rcp(new Tpetra::FECrsMatrix< val_t, local_dof_t, global_dof_t >{sparsity_graph});
    const auto glob_rhs = Teuchos::rcp(new Tpetra::FEMultiVector< val_t, local_dof_t, global_dof_t >{
        sparsity_graph->getRowMap(), sparsity_graph->getImporter(), 1});

    glob_mat->beginAssembly();
    glob_rhs->beginAssembly();
    std::pair< Eigen::Matrix< val_t, 2, 2, Eigen::RowMajor >, Eigen::Matrix< val_t, 2, 1 > > local_system;
    auto& [local_matrix, local_rhs] = local_system;
    local_matrix(0, 0)              = 1.;
    local_matrix(0, 1)              = -1.;
    local_matrix(1, 0)              = -1.;
    local_matrix(1, 1)              = 1.;
    local_rhs[0]                    = 1.;
    local_rhs[1]                    = 1.;
    my_partition.visit([&]< ElementTypes T, el_o_t O >(const Element< T, O >& element) {
        if constexpr (T == ElementTypes::Line and O == 1)
            contributeLocalSystem< std::array{ptrdiff_t{0}} >(
                local_system, element, map, *glob_mat, *glob_rhs->getVectorNonConst(0));
    });
    glob_mat->endAssembly();
    glob_rhs->endAssembly();

    constexpr auto approx = [](val_t v1, val_t v2) {
        return std::fabs(v1 - v2) < 1e-15;
    };

    try
    {
        const auto                               n_my_rows = sparsity_graph->getNodeNumRows();
        std::vector< val_t >                     val_alloc(sparsity_graph->getGlobalNumCols());
        const Teuchos::ArrayView< val_t >        val_view{val_alloc};
        std::vector< global_dof_t >              ind_alloc(sparsity_graph->getGlobalNumCols());
        const Teuchos::ArrayView< global_dof_t > ind_view{ind_alloc};
        std::vector< val_t >                     ordered_row_vals;
        size_t                                   row_size         = 0;
        const auto                               check_matrix_row = [&](global_dof_t dof) {
            const bool           is_first          = dof == 0;
            const bool           is_last           = dof == node_dist.size() - 1;
            const size_t         expected_row_size = (is_first or is_last) ? 2 : 3;
            std::vector< val_t > expected_vals;
            if (is_first)
                expected_vals = {1., -1.};
            else if (is_last)
                expected_vals = {-1., 1.};
            else
                expected_vals = {-1., 2., -1.};

            const auto inds         = std::views::counted(ind_alloc.begin(), row_size);
            const auto vals         = std::views::counted(val_alloc.begin(), row_size);
            const auto sorting_perm = sortingPermutation(inds.begin(), inds.end());
            ordered_row_vals.resize(row_size);
            copyPermuted(vals.begin(), vals.end(), sorting_perm.begin(), ordered_row_vals.begin());

            if (row_size != expected_row_size)
            {
                std::stringstream err_stream;
                err_stream << "Incorrect row size of row " << dof << "\nExpected size: " << expected_row_size
                           << "\nActual size: " << row_size << '\n';
                const auto err_string = err_stream.str();
                throw std::logic_error{err_string.c_str()};
            }
            if (not std::ranges::equal(ordered_row_vals, expected_vals, approx))
            {
                std::stringstream err_stream;
                err_stream << "Incorrect row entries of row " << dof << "\nExpected entries: ";
                for (auto v : expected_vals)
                    err_stream << v << ' ';
                err_stream << "\nActual entries: ";
                for (auto v : ordered_row_vals)
                    err_stream << v << ' ';
                err_stream << '\n';
                const auto err_string = err_stream.str();
                throw std::logic_error{err_string.c_str()};
            }
        };

        const auto local_rhs_view  = glob_rhs->getData(0);
        const auto check_rhs_entry = [&](size_t local_ind, global_dof_t global_ind) {
            const double expected = (global_ind == 0 or global_ind == node_dist.size() - 1) ? 1. : 2.;
            if (not approx(local_rhs_view[local_ind], expected))
                throw std::logic_error{"Incorrect RHS entry"};
        };

        for (size_t local_row = 0; local_row < n_my_rows; ++local_row)
        {
            const auto global_row = sparsity_graph->getRowMap()->getGlobalElement(local_row);
            glob_mat->getGlobalRowCopy(global_row, ind_view, val_view, row_size);
            check_matrix_row(global_row);
            check_rhs_entry(local_row, global_row);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        comm.abort();
    }
}
