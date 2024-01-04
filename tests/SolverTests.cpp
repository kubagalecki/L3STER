#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/solve/Amesos2Solvers.hpp"
#include "l3ster/solve/BelosSolvers.hpp"

#include "Tpetra_Core.hpp"

#include <numeric>

using namespace lstr;

// A view of Tpetra::CrsMatrix as a Tpetra::Operator outside the inheritance hierarchy
class TpetraMatrixAsOperatorStrongUpcast : virtual public util::DiagonalAwareOperator
{
public:
    TpetraMatrixAsOperatorStrongUpcast(const Teuchos::RCP< const tpetra_crsmatrix_t >& matrix) : m_matrix{matrix} {}

    Teuchos::RCP< const tpetra_map_t > getDomainMap() const override { return m_matrix->getDomainMap(); }
    Teuchos::RCP< const tpetra_map_t > getRangeMap() const override { return m_matrix->getRangeMap(); }
    bool                               hasTransposeApply() const override { return m_matrix->hasTransposeApply(); }
    void                               apply(const tpetra_multivector_t& x,
                                             tpetra_multivector_t&       y,
                                             Teuchos::ETransp            trans,
                                             val_t                       alpha,
                                             val_t                       beta) const override
    {
        m_matrix->apply(x, y, trans, alpha, beta);
    }

    Teuchos::RCP< tpetra_vector_t > initDiagonalCopy() const override
    {
        return util::makeTeuchosRCP< tpetra_vector_t >(m_matrix->getRowMap(), false);
    }
    void fillDiagonalCopy(tpetra_vector_t& diag) const override { m_matrix->getLocalDiagCopy(diag); };

private:
    Teuchos::RCP< const tpetra_crsmatrix_t > m_matrix;
};

// Make 1D diffusion matrix
auto makeSPDMatrix(size_t size) -> Teuchos::RCP< const lstr::tpetra_crsmatrix_t >
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

auto equalMV(const tpetra_multivector_t& v1, const tpetra_multivector_t& v2) -> std::vector< val_t >
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
    return col_norms;
}

bool checkNorms(const std::vector< val_t >& norms, std::string_view test_name, val_t tol = 1e-6)
{
    const auto pass = std::ranges::all_of(norms, [tol](val_t n) { return n <= tol; });
    int        rank{};
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    if (rank == 0)
    {
        std::cerr << (pass ? "Passed" : "Failed") << " test:\t\t\t" << test_name << '\n' << "With error norms:\t";
        for (auto n : norms)
            std::cerr << n << '\t';
        std::cerr << "\n\n";
    }
    return pass;
}

bool lapackTest(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                const Teuchos::RCP< tpetra_multivector_t >&       x,
                const Teuchos::RCP< const tpetra_multivector_t >& b,
                const tpetra_multivector_t&                       solution)
{
    auto solver = solvers::Lapack{};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "Lapack direct solver test", 1e-10);
}

bool klu2Test(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
              const Teuchos::RCP< tpetra_multivector_t >&       x,
              const Teuchos::RCP< const tpetra_multivector_t >& b,
              const tpetra_multivector_t&                       solution)
{
    auto solver = solvers::KLU2{};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "KLU2 direct solver test", 1e-10);
}

bool cgNoprecTest(const Teuchos::RCP< const tpetra_operator_t >&    A,
                  const Teuchos::RCP< tpetra_multivector_t >&       x,
                  const Teuchos::RCP< const tpetra_multivector_t >& b,
                  const tpetra_multivector_t&                       solution)
{
    auto solver = CG{{.verbosity{.summary = false}}};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "Unpreconditioned CG solver test", 1e-5);
}

bool cgRichardsonTest(const Teuchos::RCP< const tpetra_operator_t >&    A,
                      const Teuchos::RCP< tpetra_multivector_t >&       x,
                      const Teuchos::RCP< const tpetra_multivector_t >& b,
                      const tpetra_multivector_t&                       solution)
{
    const auto solver_opts  = IterSolverOpts{.verbosity = {.summary = false}};
    auto       precond_opts = RichardsonOpts{.damping = .5};
    auto       solver       = CG{solver_opts, precond_opts};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "Richardson preconditioner + CG solver test", 1e-5);
}

bool cgJacobiTest(const Teuchos::RCP< const tpetra_operator_t >&    A,
                  const Teuchos::RCP< tpetra_multivector_t >&       x,
                  const Teuchos::RCP< const tpetra_multivector_t >& b,
                  const tpetra_multivector_t&                       solution)
{
    const auto solver_opts  = IterSolverOpts{.verbosity = {.summary = false}};
    auto       precond_opts = JacobiOpts{.damping = .5};
    auto       solver       = CG{solver_opts, precond_opts};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "Jacobi preconditioner + CG solver test", 1e-5);
}

bool cgSGSTest(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
               const Teuchos::RCP< tpetra_multivector_t >&       x,
               const Teuchos::RCP< const tpetra_multivector_t >& b,
               const tpetra_multivector_t&                       solution)
{
    const auto solver_opts  = IterSolverOpts{.verbosity = {.summary = false}};
    auto       precond_opts = SGSOpts{.damping = .5};
    auto       solver       = CG{solver_opts, precond_opts};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "Ifpack2 symmetric Gauss-Seidl preconditioner + CG solver test", 1e-5);
}

bool cgChebyshevTest(const Teuchos::RCP< const tpetra_crsmatrix_t >&   A,
                     const Teuchos::RCP< tpetra_multivector_t >&       x,
                     const Teuchos::RCP< const tpetra_multivector_t >& b,
                     const tpetra_multivector_t&                       solution)
{
    const auto solver_opts  = IterSolverOpts{.verbosity = {.summary = false}};
    auto       precond_opts = ChebyshevOpts{.degree = 3};
    auto       solver       = CG{solver_opts, precond_opts};
    solver.solve(A, b, x);
    const auto norms = equalMV(solution, *x);
    x->putScalar(0.);
    return checkNorms(norms, "Ifpack2 3rd order Chebyshev preconditioner + CG solver test", 1e-5);
}

int main(int argc, char* argv[])
{
#ifdef _OPENMP
    omp_set_num_threads(2); // avoid oversubscription (for ncores >= 8), but still thread-parallel
#endif
    Tpetra::ScopeGuard tpetraScope(&argc, &argv);

    constexpr size_t size  = 100;
    constexpr size_t n_rhs = 2;

    const auto A      = makeSPDMatrix(size);
    const auto x      = util::makeTeuchosRCP< tpetra_multivector_t >(A->getRowMap(), n_rhs);
    const auto b      = util::makeTeuchosRCP< tpetra_multivector_t >(A->getRowMap(), n_rhs);
    const auto A_mfop = util::makeTeuchosRCP< TpetraMatrixAsOperatorStrongUpcast >(A);

    x->randomize();
    const auto solution = createCopy(*x);

    A->apply(*x, *b);
    x->putScalar(0.);

    bool pass = true;

    // Direct solvers
    pass &= lapackTest(A, x, b, solution);
    pass &= A->getComm()->getSize() != 1 ? true : klu2Test(A, x, b, solution); // KLU2 fails when nranks > 1 :(

    // CG with Ifpack2 preconditioners
    pass &= cgNoprecTest(A, x, b, solution);
    pass &= cgRichardsonTest(A, x, b, solution);
    pass &= cgJacobiTest(A, x, b, solution);
    pass &= cgSGSTest(A, x, b, solution);
    pass &= cgChebyshevTest(A, x, b, solution);

    // CG with native preconditioners
    pass &= cgNoprecTest(A_mfop, x, b, solution);
    pass &= cgRichardsonTest(A_mfop, x, b, solution);
    pass &= cgJacobiTest(A_mfop, x, b, solution);

    return pass ? EXIT_SUCCESS : EXIT_FAILURE;
}