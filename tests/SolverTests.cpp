#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/solve/Amesos2Solvers.hpp"
#include "l3ster/solve/BelosSolvers.hpp"
#include "l3ster/solve/Ifpack2Preconditioners.hpp"
#include "l3ster/solve/NativePreconditioners.hpp"

#include <numeric>

#include <print>

using namespace lstr;
using namespace lstr::solvers;

// Make 1D diffusion matrix
auto makeSPDMatrix(size_t size) -> Teuchos::RCP< const tpetra_crsmatrix_t >
{
    auto       comm     = util::makeTeuchosRCP< Teuchos::MpiComm< int > >(MPI_COMM_WORLD);
    const auto row_map  = util::makeTeuchosRCP< tpetra_map_t >(size, 0, comm);
    const auto n_diag   = row_map->getLocalNumElements();
    const auto graph    = util::makeTeuchosRCP< tpetra_crsgraph_t >(row_map, 3u);
    auto       col_dofs = std::array< global_dof_t, 3 >{};
    for (size_t i = 0; i != n_diag; ++i)
    {
        const auto global_row = row_map->getGlobalElement(static_cast< local_dof_t >(i));
        if (global_row == 0)
        {
            col_dofs[0] = 0;
            col_dofs[1] = 1;
            graph->insertGlobalIndices(0, Teuchos::ArrayView{col_dofs.data(), 2});
        }
        else if (static_cast< size_t >(global_row) == size - 1)
        {
            col_dofs[0] = global_row - 1;
            col_dofs[1] = global_row;
            graph->insertGlobalIndices(global_row, Teuchos::ArrayView{col_dofs.data(), 2});
        }
        else
        {
            std::iota(col_dofs.begin(), col_dofs.end(), global_row - 1);
            graph->insertGlobalIndices(global_row, Teuchos::ArrayView{col_dofs.data(), col_dofs.size()});
        }
    }
    graph->fillComplete();
    const auto matrix = util::makeTeuchosRCP< tpetra_crsmatrix_t >(graph);

    for (size_t i = 0; i != n_diag; ++i)
    {
        const auto global_row = row_map->getGlobalElement(static_cast< local_dof_t >(i));
        if (global_row == 0)
        {
            col_dofs[0]     = 0;
            col_dofs[1]     = 1;
            const auto vals = std::array{1., 0.};
            matrix->replaceGlobalValues(
                global_row, Teuchos::ArrayView{col_dofs.data(), 2}, Teuchos::ArrayView{vals.data(), vals.size()});
        }
        if (global_row == 1)
        {
            std::iota(col_dofs.begin(), col_dofs.end(), global_row - 1);
            const auto vals = std::array{0., 2., .5};
            matrix->replaceGlobalValues(global_row,
                                        Teuchos::ArrayView{col_dofs.data(), col_dofs.size()},
                                        Teuchos::ArrayView{vals.data(), vals.size()});
        }
        else if (static_cast< size_t >(global_row) == size - 2)
        {
            std::iota(col_dofs.begin(), col_dofs.end(), global_row - 1);
            const auto vals = std::array{.5, 2., 0.};
            matrix->replaceGlobalValues(global_row,
                                        Teuchos::ArrayView{col_dofs.data(), col_dofs.size()},
                                        Teuchos::ArrayView{vals.data(), vals.size()});
        }
        else if (static_cast< size_t >(global_row) == size - 1)
        {
            col_dofs[0]     = global_row - 1;
            col_dofs[1]     = global_row;
            const auto vals = std::array{0., 1.};
            matrix->replaceGlobalValues(
                global_row, Teuchos::ArrayView{col_dofs.data(), 2}, Teuchos::ArrayView{vals.data(), vals.size()});
        }
        else
        {
            const auto vals = std::array{.5, 2., .5};
            std::iota(col_dofs.begin(), col_dofs.end(), global_row - 1);
            matrix->replaceGlobalValues(global_row,
                                        Teuchos::ArrayView{col_dofs.data(), col_dofs.size()},
                                        Teuchos::ArrayView{vals.data(), vals.size()});
        }
    }
    matrix->fillComplete();
    return matrix;
}

#ifdef L3STER_TRILINOS_HAS_AMESOS2
auto diffMVNorm(const tpetra_multivector_t& v1, const tpetra_multivector_t& v2) -> val_t
{
    auto diff = createCopy(v1);
    {
        auto view1     = v1.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto view2     = v2.getLocalViewHost(Tpetra::Access::ReadOnly);
        auto diff_view = diff.getLocalViewHost(Tpetra::Access::OverwriteAll);

        for (size_t i = 0; i != diff_view.extent(0); ++i)
            for (size_t j = 0; j != diff_view.extent(1); ++j)
                diff_view(i, j) = view1(i, j) - view2(i, j);
    }

    auto col_norms = std::vector< val_t >(diff.getNumVectors());
    diff.norm2(col_norms);
    return std::sqrt(
        std::transform_reduce(col_norms.begin(), col_norms.end(), 0., std::plus{}, [](auto n) { return n * n; }));
}

void printDirectHeader()
{
    std::println("{:-^73}\n{:^15}{:^12}{:^10}", " DIRECT SOLVER TESTS ", "Name", "Error", "Status");
}

template < DirectSolver_c Solver >
bool runDirectTest(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                   const Teuchos::RCP< tpetra_multivector_t >&       x,
                   const Teuchos::RCP< const tpetra_multivector_t >& b,
                   const tpetra_multivector_t&                       solution,
                   std::string_view                                  name)
{
    auto solver = Solver{};
    x->putScalar(0.);
    solver.solve(A, b, x);
    const auto err_norm = diffMVNorm(*x, solution);
    const bool passed   = err_norm <= 1e-8;
    if (A->getComm()->getRank() == 0)
        std::println("{:<15}{:^12.3e}{:^10}", name, err_norm, passed ? "PASS" : "FAIL");
    return passed;
}

bool directSolverSuite(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                       const Teuchos::RCP< tpetra_multivector_t >&       x,
                       const Teuchos::RCP< const tpetra_multivector_t >& b,
                       const tpetra_multivector_t&                       solution)
{
    bool passed = true;
    if (A->getComm()->getRank() == 0)
        printDirectHeader();

    if (A->getComm()->getSize() == 1) // KLU2 fails when nranks > 1 :(
        passed &= runDirectTest< Klu2 >(A, x, b, solution, "Amesos2::KLU2");
    passed &= runDirectTest< Lapack >(A, x, b, solution, "Amesos2::Lapack");

    if (A->getComm()->getRank() == 0)
        std::println();
    return passed;
}

#else
bool directSolverSuite(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                       const Teuchos::RCP< tpetra_multivector_t >&       x,
                       const Teuchos::RCP< const tpetra_multivector_t >& b,
                       const tpetra_multivector_t&                       solution)
{
    return true;
}
#endif

#ifdef L3STER_TRILINOS_HAS_BELOS
void printIterHeader()
{
    std::println(
        "{:-^73}\n{:^37}{:^12}{:^16}{:^10}", " ITERATIVE SOLVER TESTS ", "Name", "Error", "Iterations", "Status");
}

template < PreconditionerOptions_c Opts = NullPreconditioner::Options >
bool runCGTest(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
               const Teuchos::RCP< tpetra_multivector_t >&       x,
               const Teuchos::RCP< const tpetra_multivector_t >& b,
               std::string_view                                  name,
               const Opts&                                       precond_opts = {})
{
    const auto solver_opts = IterSolverOpts{.verbosity = {.summary = false}};
    auto       solver      = CG{solver_opts, precond_opts};
    x->putScalar(0.);
    const auto [err_norm, iters] = solver.solve(A, b, x);
    const bool passed            = err_norm <= IterSolverOpts{}.tol;
    if (A->getComm()->getRank() == 0)
        std::println("{:<37}{:^12.3e}{:^16}{:^10}", name, err_norm, iters, passed ? "PASS" : "FAIL");
    return passed;
}

template < PreconditionerOptions_c Opts = NullPreconditioner::Options >
bool runGMRESTest(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                  const Teuchos::RCP< tpetra_multivector_t >&       x,
                  const Teuchos::RCP< const tpetra_multivector_t >& b,
                  std::string_view                                  name,
                  const Opts&                                       precond_opts = {})
{
    const auto solver_opts = IterSolverOpts{.verbosity = {.summary = false}};
    auto       solver      = Gmres{solver_opts, precond_opts};
    x->putScalar(0.);
    const auto [err_norm, iters] = solver.solve(A, b, x);
    const bool passed            = err_norm <= IterSolverOpts{}.tol;
    if (A->getComm()->getRank() == 0)
        std::println("{:<37}{:^12.3e}{:^16}{:^10}", name, err_norm, iters, passed ? "PASS" : "FAIL");
    return passed;
}

bool iterativeSolverSuite(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                          const Teuchos::RCP< tpetra_multivector_t >&       x,
                          const Teuchos::RCP< const tpetra_multivector_t >& b)
{
    bool passed = true;
    if (A->getComm()->getRank() == 0)
        printIterHeader();

    passed &= runCGTest(A, x, b, "CG without precond.");
    passed &= runCGTest(A, x, b, "CG + native Richardson precond.", NativeRichardsonOpts{});
    passed &= runCGTest(A, x, b, "CG + native Jacobi precond.", NativeJacobiOpts{});

    passed &= runCGTest(A, x, b, "GMRES without precond.");
    passed &= runCGTest(A, x, b, "GMRES + native Richardson precond.", NativeRichardsonOpts{});
    passed &= runCGTest(A, x, b, "GMRES + native Jacobi precond.", NativeJacobiOpts{});

#ifdef L3STER_TRILINOS_HAS_IFPACK2
    passed &= passed &= runCGTest(A, x, b, "CG + Ifpack2 Richardson precond.", Ifpack2RichardsonOpts{});
    passed &= passed &= runCGTest(A, x, b, "CG + Ifpack2 Jacobi precond.", Ifpack2JacobiOpts{});
    passed &= passed &= runCGTest(A, x, b, "CG + Ifpack2 SGS precond.", Ifpack2SGSOpts{});
    passed &= passed &= runCGTest(A, x, b, "CG + Ifpack2 Chebyshev precond.", Ifpack2ChebyshevOpts{});

    passed &= passed &= runGMRESTest(A, x, b, "GMRES + Ifpack2 Richardson precond.", Ifpack2RichardsonOpts{});
    passed &= passed &= runGMRESTest(A, x, b, "GMRES + Ifpack2 Jacobi precond.", Ifpack2JacobiOpts{});
    passed &= passed &= runGMRESTest(A, x, b, "GMRES + Ifpack2 SGS precond.", Ifpack2SGSOpts{});
    passed &= passed &= runGMRESTest(A, x, b, "GMRES + Ifpack2 Chebyshev precond.", Ifpack2ChebyshevOpts{});
    passed &= passed &= runGMRESTest(A, x, b, "GMRES + Ifpack2 ILUk precond.", Ifpack2IluKOpts{});
    passed &= passed &= runGMRESTest(A, x, b, "GMRES + Ifpack2 ILUT precond.", Ifpack2IluTOpts{});
#endif

    if (A->getComm()->getRank() == 0)
        std::println();
    return passed;
}
#else
bool iterativeSolverSuite(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                          const Teuchos::RCP< tpetra_multivector_t >&       x,
                          const Teuchos::RCP< const tpetra_multivector_t >& b)
{
    return true;
}
#endif

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    omp_set_num_threads(2); // avoid oversubscription (for ncores >= 8), but still thread-parallel
#endif
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    constexpr size_t size  = 100;
    constexpr size_t n_rhs = 2;

    const auto A = makeSPDMatrix(size);
    const auto x = util::makeTeuchosRCP< tpetra_multivector_t >(A->getRowMap(), n_rhs);
    const auto b = util::makeTeuchosRCP< tpetra_multivector_t >(A->getRowMap(), n_rhs);

    x->randomize();
    const auto solution = createCopy(*x);

    A->apply(*x, *b);
    x->putScalar(0.);

    bool pass = true;

    // Direct solvers
    pass &= directSolverSuite(A, x, b, solution);
    pass &= iterativeSolverSuite(A, x, b);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}