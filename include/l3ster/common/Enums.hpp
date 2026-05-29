#ifndef L3STER_COMMON_ENUMS_HPP
#define L3STER_COMMON_ENUMS_HPP

namespace lstr
{
enum struct Space
{
    X = 0,
    Y = 1,
    Z = 2
};

enum struct CondensationPolicy
{
    None,
    ElementBoundary
};

enum struct OperatorEvaluationStrategy
{
    GlobalAssembly,
    MatrixFree
};
} // namespace lstr
#endif // L3STER_COMMON_ENUMS_HPP
