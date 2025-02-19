#include "AdvectionPeriodic2D.hpp"

int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{4};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    solveAdvection2D< OperatorEvaluationStrategy::MatrixFree, CondensationPolicy::None >();
}
