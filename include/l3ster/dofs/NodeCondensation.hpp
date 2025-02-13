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
} // namespace dofs
} // namespace lstr
#endif // L3STER_DOFS_NODECONDENSATION_HPP
