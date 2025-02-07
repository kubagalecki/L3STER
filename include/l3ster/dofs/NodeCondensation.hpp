#ifndef L3STER_DOFS_NODECONDENSATION_HPP
#define L3STER_DOFS_NODECONDENSATION_HPP

#include "l3ster/bcs/PeriodicBC.hpp"
#include "l3ster/comm/ImportExport.hpp"
#include "l3ster/common/Enums.hpp"
#include "l3ster/common/ProblemDefinition.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

namespace lstr
{
template < CondensationPolicy cond_policy >
struct CondensationPolicyTag
{
    static constexpr auto value = cond_policy;
};

inline constexpr auto no_condensation_tag  = CondensationPolicyTag< CondensationPolicy::None >{};
inline constexpr auto element_boundary_tag = CondensationPolicyTag< CondensationPolicy::ElementBoundary >{};

namespace dofs
{
template < CondensationPolicy CP, mesh::ElementType ET, el_o_t EO >
constexpr decltype(auto) getPrimaryNodesView(const mesh::Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
        return getBoundaryNodes(element);
}

template < CondensationPolicy CP, mesh::ElementType ET, el_o_t EO >
constexpr decltype(auto) getPrimaryNodesArray(const mesh::Element< ET, EO >& element, CondensationPolicyTag< CP > = {})
{
    if constexpr (CP == CondensationPolicy::None)
        return element.getNodes();
    else if constexpr (CP == CondensationPolicy::ElementBoundary)
    {
        std::array< n_id_t, mesh::ElementTraits< mesh::Element< ET, EO > >::boundary_node_inds.size() > retval;
        std::ranges::copy(getBoundaryNodes(element), retval.begin());
        return retval;
    }
}

template < CondensationPolicy CP, mesh::ElementType ET, el_o_t EO >
consteval auto getNumPrimaryNodes(util::ValuePack< CP, ET, EO > = {}) -> size_t
{
    return std::tuple_size_v<
        std::decay_t< decltype(getPrimaryNodesArray< CP >(std::declval< mesh::Element< ET, EO > >())) > >;
};

template < CondensationPolicy CP, ProblemDef problem_def, el_o_t... orders >
auto getActiveNodes(const mesh::MeshPartition< orders... >& mesh,
                    util::ConstexprValue< problem_def >,
                    CondensationPolicyTag< CP > = {}) -> std::vector< n_id_t >
{
    auto active_nodes_set = robin_hood::unordered_flat_set< n_id_t >{};
    mesh.visit(
        [&]< mesh::ElementType ET, el_o_t EO >(const mesh::Element< ET, EO >& element) {
            for (auto n : getPrimaryNodesView< CP >(element))
                active_nodes_set.insert(n);
        },
        problem_def | std::views::transform([](const DomainDef< problem_def.n_fields >& d) { return d.domain; }));
    auto retval = std::vector< n_id_t >{};
    retval.reserve(active_nodes_set.size());
    std::ranges::copy(active_nodes_set, std::back_inserter(retval));
    std::ranges::sort(retval);
    return retval;
}
} // namespace dofs
} // namespace lstr
#endif // L3STER_DOFS_NODECONDENSATION_HPP
