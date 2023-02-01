#ifndef L3STER_SIMULATIONGRAPH_HPP
#define L3STER_SIMULATIONGRAPH_HPP

#include "l3ster/simdef/SimulationComponents.hpp"

#include <variant>

namespace lstr::def
{
template < std::copy_constructible... Kernels >
class SimulationGraph
{
    struct Action;
    struct PredicateAction;
    using control_path_t = std::variant< const Action*, const PredicateAction* >;

    struct Action
    {
        control_path_t next;
    };
    struct PredicateAction
    {
        control_path_t next_if_true, next_if_false;
    };
};
} // namespace lstr::def
#endif // L3STER_SIMULATIONGRAPH_HPP
