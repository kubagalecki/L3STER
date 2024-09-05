#ifndef L3STER_COMMON_ENUMS_HPP
#define L3STER_COMMON_ENUMS_HPP

namespace lstr
{
enum struct Space
{
    X,
    Y,
    Z
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
