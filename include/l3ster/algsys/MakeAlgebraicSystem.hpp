#ifndef L3STER_ALGSYS_MAKEALGEBRAICSYSTEM_HPP
#define L3STER_ALGSYS_MAKEALGEBRAICSYSTEM_HPP

#include "l3ster/algsys/AssembledSystem.hpp"
#include "l3ster/algsys/MatrixFreeSystem.hpp"

namespace lstr
{
template < el_o_t... orders, size_t max_dofs_per_node, AlgebraicSystemParams algsys_params = {} >
auto makeAlgebraicSystem(std::shared_ptr< const MpiComm >                          comm,
                         std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                         const ProblemDefinition< max_dofs_per_node >&             problem_def,
                         const BCDefinition< max_dofs_per_node >&                  bc_def = {},
                         util::ConstexprValue< algsys_params >                            = {})
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto n_rhs = algsys_params.n_rhs;
    if constexpr (algsys_params.eval_strategy == OperatorEvaluationStrategy::GlobalAssembly)
    {
        constexpr auto cond_policy = algsys_params.cond_policy;
        return algsys::AssembledSystem< max_dofs_per_node, cond_policy, n_rhs, orders... >{
            std::move(comm), std::move(mesh), problem_def, bc_def};
    }
    else
    {
        static_assert(algsys_params.eval_strategy == OperatorEvaluationStrategy::MatrixFree);
        static_assert(algsys_params.cond_policy == CondensationPolicy::None,
                      "Matrix-free operator evaluation and static condensation are mutually exclusive");
        return algsys::MatrixFreeSystem< max_dofs_per_node, n_rhs, orders... >{
            std::move(comm), std::move(mesh), problem_def, bc_def};
    }
}

template < el_o_t... orders, size_t max_dofs_per_node, AlgebraicSystemParams algsys_params = {} >
auto makeAlgebraicSystem(std::shared_ptr< const MpiComm >                    comm,
                         std::shared_ptr< mesh::MeshPartition< orders... > > mesh,
                         const ProblemDefinition< max_dofs_per_node >&       problem_def,
                         const BCDefinition< max_dofs_per_node >&            bc_def        = {},
                         util::ConstexprValue< algsys_params >               params_ctwrpr = {})
{
    return makeAlgebraicSystem(std::move(comm),
                               std::shared_ptr< const mesh::MeshPartition< orders... > >{std::move(mesh)},
                               problem_def,
                               bc_def,
                               params_ctwrpr);
}

template < el_o_t... orders,
           ProblemDef            problem_def,
           ProblemDef            dirichlet_def = ProblemDef< 0, problem_def.n_fields >{},
           AlgebraicSystemParams algsys_params = {} >
[[deprecated]] auto makeAlgebraicSystem(std::shared_ptr< const MpiComm >                          comm,
                                        std::shared_ptr< const mesh::MeshPartition< orders... > > mesh,
                                        util::ConstexprValue< problem_def >,
                                        util::ConstexprValue< dirichlet_def >               = {},
                                        util::ConstexprValue< algsys_params > params_ctwrpr = {})
{
    const auto prob_def = detail::convertToRuntime(problem_def);
    const auto bc_def   = algsys::detail::convertToRuntimeBC(dirichlet_def);
    return makeAlgebraicSystem(std::move(comm), std::move(mesh), prob_def, bc_def, params_ctwrpr);
}

template < el_o_t... orders,
           ProblemDef            problem_def,
           ProblemDef            dirichlet_def = ProblemDef< 0, problem_def.n_fields >{},
           AlgebraicSystemParams params        = {} >
[[deprecated]] auto makeAlgebraicSystem(std::shared_ptr< const MpiComm >                    comm,
                                        std::shared_ptr< mesh::MeshPartition< orders... > > mesh,
                                        util::ConstexprValue< problem_def >                 problemdef_ctwrpr,
                                        util::ConstexprValue< dirichlet_def >               dbcdef_ctwrpr = {},
                                        util::ConstexprValue< params >                      params_ctwrpr = {})
{
    return makeAlgebraicSystem(std::move(comm),
                               std::shared_ptr< const mesh::MeshPartition< orders... > >{std::move(mesh)},
                               problemdef_ctwrpr,
                               dbcdef_ctwrpr,
                               params_ctwrpr);
}
} // namespace lstr
#endif // L3STER_ALGSYS_MAKEALGEBRAICSYSTEM_HPP
