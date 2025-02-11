#include "Diffusion3D.hpp"

int main(int argc, char* argv[])
{
    L3sterScopeGuard scope_guard{argc, argv};
    solveDiffusion3DProblem< CondensationPolicy::ElementBoundary, OperatorEvaluationStrategy::GlobalAssembly >();
}