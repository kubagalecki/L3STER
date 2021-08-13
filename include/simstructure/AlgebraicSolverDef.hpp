#ifndef L3STER_SIMSTRUCTURE_ALGEBRAICSOLVERDEF_HPP
#define L3STER_SIMSTRUCTURE_ALGEBRAICSOLVERDEF_HPP

namespace lstr::def
{
enum struct AlgebraicSolverTypes
{
    DirectSuperLU,
    DirectMUMPS,
    AMG_PCG
};

template < AlgebraicSolverTypes T >
struct AlgebraicSolver
{};

template <>
struct AlgebraicSolver< AlgebraicSolverTypes::AMG_PCG >
{
    int    levels;
    double tol;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_ALGEBRAICSOLVERDEF_HPP
