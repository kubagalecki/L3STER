#include "Diffusion2D.hpp"

// Solve 2D diffusion problem
int main(int argc, char* argv[])
{
    const auto max_par_guard = util::MaxParallelismGuard{3};
    const auto scope_guard   = L3sterScopeGuard{argc, argv};
    test< CondensationPolicy::None, OperatorEvaluationStrategy::MatrixFree >();
}
