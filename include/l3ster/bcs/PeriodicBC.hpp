#ifndef L3STER_BCS_PERIODICBC_HPP
#define L3STER_BCS_PERIODICBC_HPP

#include "l3ster/bcs/BCDefinition.hpp"
#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mapping/MapReferenceToPhysical.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/mesh/NodeReferenceLocation.hpp"
#include "l3ster/util/Caliper.hpp"
#include "l3ster/util/IndexMap.hpp"
#include "l3ster/util/SpatialHashTable.hpp"

#include <array>
#include <bitset>

namespace lstr::bcs
{
template < size_t max_dofs_per_node >
class PeriodicBC
{
    using description_t = std::array< n_id_t, max_dofs_per_node >;

public:
    PeriodicBC() = default;
    template < el_o_t... orders >
    PeriodicBC(const PeriodicBCDefinition< max_dofs_per_node >& definition,
               const mesh::MeshPartition< orders... >&          mesh,
               const MpiComm&                                   comm);

    auto begin() const { return m_periodic_info.begin(); }
    auto end() const { return m_periodic_info.end(); }
    auto size() const { return m_periodic_info.size(); }
    auto lookup(n_id_t node) const -> description_t
    {
        const auto iter = m_periodic_info.find(node);
        return iter == m_periodic_info.end() ? util::makeFilledArray< max_dofs_per_node >(invalid_node) : iter->second;
    }
    auto getPeriodicGhosts() const -> std::span< const n_id_t > { return m_periodic_ghosts; }

private:
    robin_hood::unordered_flat_map< n_id_t, description_t > m_periodic_info;
    util::ArrayOwner< n_id_t >                              m_periodic_ghosts;
};
template < el_o_t... orders, size_t max_dofs_per_node >
PeriodicBC(const PeriodicBCDefinition< max_dofs_per_node >&, const mesh::MeshPartition< orders... >&, const MpiComm&)
    -> PeriodicBC< max_dofs_per_node >;

namespace detail
{
struct BoundaryGroup
{
    std::vector< d_id_t >  boundary_ids_src, boundary_ids_dest;
    std::array< val_t, 3 > translation;

    bool operator==(const BoundaryGroup&) const  = default;
    auto operator<=>(const BoundaryGroup&) const = default;
};
template < size_t max_dofs_per_node >
auto processPeriodicDef(const PeriodicBCDefinition< max_dofs_per_node >& def)
    -> util::ArrayOwner< std::pair< std::bitset< max_dofs_per_node >, util::ArrayOwner< BoundaryGroup > > >
{
    using dof_bmp_t           = std::bitset< max_dofs_per_node >;
    auto           dof_groups = std::array< std::vector< BoundaryGroup >, max_dofs_per_node >{};
    constexpr auto to_vec     = []< typename T >(const util::ArrayOwner< T >& a) {
        return std::vector(a.begin(), a.end());
    };
    for (const auto& [b1, b2, translation, dofs] : def)
        for (size_t i = 0; i != max_dofs_per_node; ++i)
            if (dofs.test(i))
                dof_groups[i].push_back({to_vec(b1), to_vec(b2), translation});
    for (auto& g : dof_groups)
        std::ranges::sort(g);
    auto group2dofs = std::map< std::vector< BoundaryGroup >, dof_bmp_t >{};
    for (auto&& [dof, group] : dof_groups | std::views::enumerate)
        group2dofs[group].set(dof);
    return util::ArrayOwner< std::pair< dof_bmp_t, util::ArrayOwner< BoundaryGroup > > >(
        group2dofs | std::views::transform([&](const auto& p) { return std::make_pair(p.second, p.first); }));
}

struct NodeLocationData
{
    n_id_t                 id;
    std::array< val_t, 3 > coords;
};
template < el_o_t... orders >
auto getNodeLocationData(const mesh::MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& boundary_ids)
    -> util::ArrayOwner< NodeLocationData >
{
    auto       node_to_loc    = robin_hood::unordered_flat_map< n_id_t, std::array< val_t, 3 > >{};
    const auto write_node_loc = [&]< mesh::ElementType ET, el_o_t EO >(const mesh::BoundaryElementView< ET, EO >& bv) {
        const auto& ref_x   = mesh::getNodeLocations< ET, EO >();
        const auto& el_data = bv->getData();
        for (auto i : bv.getSideNodeInds())
        {
            const auto node_id = bv->getNodes()[i];
            if (node_to_loc.contains(node_id))
                continue;
            const auto& ref_location  = ref_x[i];
            const auto  phys_location = map::mapToPhysicalSpace(el_data, ref_location);
            node_to_loc[node_id]      = phys_location;
        }
    };
    mesh.visitBoundaries(write_node_loc, boundary_ids, std::execution::seq);
    auto retval = util::ArrayOwner< NodeLocationData >(node_to_loc.size());
    std::ranges::transform(
        node_to_loc, retval.begin(), [](const auto& pair) -> NodeLocationData { return {pair.first, pair.second}; });
    return retval;
}

template < el_o_t... orders >
auto getDestNodeData(const mesh::MeshPartition< orders... >& mesh, const util::ArrayOwner< BoundaryGroup >& groups)
{
    auto target_ids = robin_hood::unordered_flat_set< d_id_t >{};
    for (const auto& g : groups)
        for (d_id_t id : g.boundary_ids_dest)
            target_ids.insert(id);
    return getNodeLocationData(mesh, target_ids);
}

template < el_o_t... orders >
auto getSourceNodeData(const mesh::MeshPartition< orders... >& mesh, const util::ArrayOwner< BoundaryGroup >& groups)
{
    auto retval = std::vector< NodeLocationData >{};
    for (const auto& [src_ids, _, dx] : groups)
    {
        const auto loc_data = getNodeLocationData(mesh, src_ids);
        std::ranges::transform(loc_data, std::back_inserter(retval), [dx](const NodeLocationData& data) {
            return NodeLocationData{data.id, util::elwise(data.coords, dx, std::plus{})};
        });
    }
    return retval;
}

inline auto makeNodeSpacemap(const util::ArrayOwner< NodeLocationData >& node_data)
    -> util::SpatialHashTable< n_id_t, 3 >
{
    if (node_data.empty())
        return util::SpatialHashTable< n_id_t, 3 >{};

    // Set up the grid with reasonable spacing and origin
    constexpr auto min_init    = util::makeFilledArray< 3 >(std::numeric_limits< val_t >::max());
    constexpr auto max_init    = util::makeFilledArray< 3 >(std::numeric_limits< val_t >::min());
    constexpr auto reduce_init = std::make_pair(min_init, max_init);
    auto coords = node_data | std::views::transform([](const auto& nd) { return nd.coords; }) | std::views::common;
    constexpr auto reduce = [](const auto& pair1, const auto& pair2) {
        const auto& [min1, max1] = pair1;
        const auto& [min2, max2] = pair2;
        return std::make_pair(util::elwise(min1, min2, util::Min{}), util::elwise(max1, max2, util::Max{}));
    };
    constexpr auto transform = [](const auto& p) {
        return std::make_pair(p, p);
    };
    const auto [min, max] = std::transform_reduce(coords.begin(), coords.end(), reduce_init, reduce, transform);
    const auto x_span     = util::elwise(max, min, std::minus{});
    const auto is_planar  = std::ranges::any_of(x_span, [](val_t d) { return d < 1.e-15; });
    const auto sz_fp      = static_cast< val_t >(node_data.size());
    const auto grid_divs  = std::round(is_planar ? std::sqrt(sz_fp) : std::cbrt(sz_fp)) * 10.;
    const auto dx         = util::elwise(x_span, util::makeFilledArray< 3 >(grid_divs), std::divides{});
    const auto origin     = util::elwise(min, max, [](val_t a, val_t b) { return std::midpoint(a, b); });

    auto retval = util::SpatialHashTable< n_id_t, 3 >{dx, origin};
    for (const auto& [id, x] : node_data)
        retval.insert(x, id);
    return retval;
}

class NodeMatcher
{
public:
    NodeMatcher(const util::ArrayOwner< NodeLocationData >& node_data, val_t tolerance)
        : m_space_map{makeNodeSpacemap(node_data)}, m_tolerance{tolerance}
    {}

    void match(std::span< const NodeLocationData > nodes)
    {
        constexpr auto distance = [](const std::array< val_t, 3 >& a, const std::array< val_t, 3 >& b) {
            const auto diff2 = util::elwise(util::elwise(a, b, std::minus{}), util::selfie(std::multiplies{}));
            return std::sqrt(util::reduce(diff2, std::plus{}));
        };
        for (const auto& [id, x] : nodes)
        {
            const auto match_crit = [&](const auto& p) {
                return distance(p.first, x) < m_tolerance;
            };
            auto       prox       = m_space_map.proximate(x);
            const auto match_iter = std::ranges::find_if(prox, match_crit);
            if (match_iter != prox.end())
            {
                m_match_src.push_back(id);
                m_match_dest.push_back(match_iter->second);
            }
        }
    }
    auto consumeMatched() -> std::pair< std::vector< n_id_t >, std::vector< n_id_t > >
    {
        return std::make_pair(std::exchange(m_match_src, std::vector< n_id_t >{}),
                              std::exchange(m_match_dest, std::vector< n_id_t >{}));
    }

private:
    util::SpatialHashTable< n_id_t, 3 > m_space_map;
    std::vector< n_id_t >               m_match_src, m_match_dest;
    val_t                               m_tolerance;
};

inline auto gatherMatched(const MpiComm& comm, NodeMatcher matcher)
{
    constexpr int src_tag = 0, dest_tag = 1;
    auto          matched = matcher.consumeMatched();
    auto& [src, dest]     = matched;
    if (comm.getRank() == 0)
    {
        const auto comm_sz = static_cast< size_t >(comm.getSize());
        auto       sizes   = util::ArrayOwner< size_t >(comm_sz + 1);
        sizes[0]           = 0;
        sizes[1]           = src.size();
        for (size_t rank = 1; rank != comm_sz; ++rank)
            sizes[rank + 1] = static_cast< size_t >(comm.probe(static_cast< int >(rank), src_tag).numElems< n_id_t >());
        std::inclusive_scan(sizes.begin(), sizes.end(), sizes.begin());
        src.resize(sizes.back());
        dest.resize(sizes.back());
        auto reqs = std::vector< MpiComm::Request >{};
        reqs.reserve(2 * comm_sz - 2);
        for (int rank = 1; rank != comm.getSize(); ++rank)
        {
            const auto offset = sizes[static_cast< size_t >(rank)];
            const auto sz     = sizes[static_cast< size_t >(rank) + 1] - offset;
            reqs.push_back(comm.receiveAsync(std::span{src}.subspan(offset, sz), rank, src_tag));
            reqs.push_back(comm.receiveAsync(std::span{dest}.subspan(offset, sz), rank, dest_tag));
        }
        MpiComm::Request::waitAll(reqs);
    }
    else
    {
        {
            auto src_req  = comm.sendAsync(src, 0, src_tag);
            auto dest_req = comm.sendAsync(dest, 0, dest_tag);
        }
        matched = {};
    }
    return matched;
}

inline auto getComponents(std::span< const n_id_t > src, std::span< const n_id_t > dest)
{
    auto nodes = std::vector< n_id_t >{};
    nodes.reserve(src.size() + dest.size());
    std::ranges::copy(src, std::back_inserter(nodes));
    std::ranges::copy(dest, std::back_inserter(nodes));
    util::sortRemoveDup(nodes);
    const auto g2l_map    = util::IndexMap< n_id_t, n_id_t >{nodes};
    const auto graph      = util::makeCrsGraph(src | std::views::transform(std::cref(g2l_map)),
                                          dest | std::views::transform(g2l_map),
                                          util::GraphType::Undirected);
    auto       components = util::getComponentsUndirected(graph);
    for (auto& n : components | std::views::join)
        n = nodes.at(n);
    return components;
}

template < el_o_t... orders >
auto scatterMatched(const mesh::MeshPartition< orders... >&          mesh,
                    const MpiComm&                                   comm,
                    const std::vector< util::ArrayOwner< n_id_t > >& components)
    -> std::array< std::vector< n_id_t >, 2 >
{
    constexpr int p_tag = 0, a_tag = 1;
    const auto    num_owned_nodes = mesh.getOwnedNodes().size();
    if (comm.getRank() == 0)
    {
        auto owned = util::ArrayOwner< size_t >(static_cast< size_t >(comm.getSize()));
        comm.gather(std::views::single(num_owned_nodes), owned.begin(), 0);
        std::inclusive_scan(owned.begin(), owned.end(), owned.begin());
        const auto get_owner = [&](n_id_t node) {
            return std::distance(owned.begin(), std::ranges::upper_bound(owned, node));
        };
        const auto comm_sz  = static_cast< size_t >(comm.getSize());
        auto       messages = util::ArrayOwner< std::array< std::vector< n_id_t >, 2 > >(comm_sz);
        for (const auto& c : components)
        {
            util::throwingAssert(c.size() > 1, "Unmatched node");
            const auto active_node = c.front();
            for (auto passive_node : c | std::views::drop(1))
            {
                const auto owner = get_owner(passive_node);
                auto& [p, a]     = messages.at(owner);
                p.push_back(passive_node);
                a.push_back(active_node);
            }
        }
        auto reqs = std::vector< MpiComm::Request >{};
        reqs.reserve(messages.size() * 2 - 2);
        for (const auto& [dest, payload] : messages | std::views::enumerate | std::views::drop(1))
        {
            const int dest_rank = static_cast< int >(dest);
            const auto& [p, a]  = payload;
            reqs.push_back(comm.sendAsync(p, dest_rank, p_tag));
            reqs.push_back(comm.sendAsync(a, dest_rank, a_tag));
        }
        MpiComm::Request::waitAll(reqs);
        return messages.front();
    }
    else
    {
        size_t _{};
        comm.gather(std::views::single(num_owned_nodes), &_, 0);
        auto retval        = std::array< std::vector< n_id_t >, 2 >{};
        auto& [p, a]       = retval;
        const auto recv_sz = static_cast< size_t >(comm.probe(0, p_tag).numElems< n_id_t >());
        p.resize(recv_sz);
        a.resize(recv_sz);
        const auto p_req = comm.receiveAsync(p, 0, p_tag);
        const auto a_req = comm.receiveAsync(a, 0, a_tag);
        return retval;
    }
}
} // namespace detail

template < size_t max_dofs_per_node >
template < el_o_t... orders >
PeriodicBC< max_dofs_per_node >::PeriodicBC(const PeriodicBCDefinition< max_dofs_per_node >& definition,
                                            const mesh::MeshPartition< orders... >&          mesh,
                                            const MpiComm&                                   comm)
{
    L3STER_PROFILE_FUNCTION;
    constexpr auto map_init_element = util::makeFilledArray< max_dofs_per_node >(invalid_node);
    const auto     processed_def    = detail::processPeriodicDef(definition);
    auto           periodic_ghosts  = std::set< n_id_t >{};
    for (const auto& [dof_bitset, boundary_groups] : processed_def)
    {
        const auto src_data  = detail::getSourceNodeData(mesh, boundary_groups);
        const auto src_span  = std::span{src_data};
        const auto dest_data = detail::getDestNodeData(mesh, boundary_groups);
        auto       matcher   = detail::NodeMatcher{dest_data, definition.tolerance};
        util::staggeredAllGather(comm, src_span, [&](auto data, [[maybe_unused]] int rank) { matcher.match(data); });
        const auto [s, d]     = detail::gatherMatched(comm, std::move(matcher));
        const auto components = detail::getComponents(s, d);
        const auto [p, a]     = detail::scatterMatched(mesh, comm, components);
        const auto dof_inds   = util::getTrueInds(dof_bitset);
        for (auto&& [my_node, dof_owner] : std::views::zip(p, a))
        {
            auto& value = m_periodic_info.try_emplace(my_node, map_init_element).first->second;
            for (auto i : dof_inds)
                value[i] = dof_owner;
            if (not mesh.isOwnedNode(dof_owner) and not mesh.isGhostNode(dof_owner))
                periodic_ghosts.insert(dof_owner);
        }
    }
    m_periodic_ghosts = periodic_ghosts;
}
} // namespace lstr::bcs
#endif // L3STER_BCS_PERIODICBC_HPP
