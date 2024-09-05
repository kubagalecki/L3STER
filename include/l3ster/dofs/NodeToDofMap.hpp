#ifndef L3STER_DOFS_NODETODOFMAP_HPP
#define L3STER_DOFS_NODETODOFMAP_HPP

#include "l3ster/common/TrilinosTypedefs.h"
#include "l3ster/dofs/DofIntervals.hpp"
#include "l3ster/mesh/LocalMeshView.hpp"
#include "l3ster/util/IndexMap.hpp"

namespace lstr::dofs
{
template < size_t dofs_per_node >
class NodeToGlobalDofMap
{
    using payload_t = std::array< global_dof_t, dofs_per_node >;
    struct ContiguousCaseInfo
    {
        static_assert(dofs_per_node <= (1u << (sizeof(std::uint8_t) * CHAR_BIT)));
        n_id_t                                    base_node;
        global_dof_t                              base_dof;
        std::array< std::uint8_t, dofs_per_node > dof_inds;
        std::uint8_t                              n_dofs;
    };
    using map_t = robin_hood::unordered_flat_map< n_id_t, payload_t >;

public:
    using dof_t                               = global_dof_t;
    static constexpr global_dof_t invalid_dof = -1;
    static constexpr bool         isValid(dof_t dof) { return dof != invalid_dof; }

    NodeToGlobalDofMap() = default;
    template < CondensationPolicy CP >
    NodeToGlobalDofMap(const node_interval_vector_t< dofs_per_node >& dof_intervals,
                       const NodeCondensationMap< CP >&               cond_map);

    [[nodiscard]] inline auto operator()(n_id_t node) const -> payload_t;
    [[nodiscard]] bool        isContiguous() const { return std::get_if< ContiguousCaseInfo >(std::addressof(m_data)); }

private:
    template < CondensationPolicy CP >
    bool tryInitAsContiguous(const node_interval_vector_t< dofs_per_node >& dof_ints,
                             const NodeCondensationMap< CP >&               cond_map);
    template < CondensationPolicy CP >
    void initNonContiguous(const node_interval_vector_t< dofs_per_node >& dof_intervals,
                           const NodeCondensationMap< CP >&               cond_map);

    std::variant< map_t, ContiguousCaseInfo > m_data;
};

template < size_t dofs_per_node, size_t num_maps >
class NodeToLocalDofMap
{
    using payload_t = std::array< std::array< local_dof_t, dofs_per_node >, num_maps >;
    using map_t     = robin_hood::unordered_flat_map< n_id_t, payload_t >;

public:
    using dof_t                        = local_dof_t;
    static constexpr dof_t invalid_dof = -1;
    static constexpr bool  isValid(dof_t dof) { return dof != invalid_dof; }

    NodeToLocalDofMap() = default;
    template < CondensationPolicy CP >
    NodeToLocalDofMap(const NodeCondensationMap< CP >&           cond_map,
                      const NodeToGlobalDofMap< dofs_per_node >& global_map,
                      const std::same_as< tpetra_map_t > auto&... local_global_maps)
        requires(sizeof...(local_global_maps) == num_maps);
    [[nodiscard]] const payload_t& operator()(n_id_t node) const noexcept { return m_map.at(node); }

    [[nodiscard]] auto size() const -> size_t { return m_map.size(); }
    [[nodiscard]] auto begin() const { return m_map.cbegin(); }
    [[nodiscard]] auto end() const { return m_map.cend(); }

private:
    map_t m_map;
};

template < CondensationPolicy CP, size_t dofs_per_node >
NodeToLocalDofMap(const NodeCondensationMap< CP >&           cond_map,
                  const NodeToGlobalDofMap< dofs_per_node >& global_map,
                  const std::same_as< tpetra_map_t > auto&... local_global_maps)
    -> NodeToLocalDofMap< dofs_per_node, sizeof...(local_global_maps) >;

template < size_t max_dofs_per_node >
class LocalDofMap
{
public:
    using dof_t                        = local_dof_t;
    static constexpr dof_t invalid_dof = -1;
    static constexpr bool  isValid(dof_t dof) { return dof != invalid_dof; }

    LocalDofMap() = default;
    LocalDofMap(const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                const NodeToGlobalDofMap< max_dofs_per_node >&         global_map,
                const mesh::NodeMap&                                   node_map,
                std::span< const global_dof_t >                        all_dofs,
                size_t                                                 num_owned_dofs);

    [[nodiscard]] auto   operator()(n_loc_id_t node) const -> const auto& { return m_map.at(node); }
    [[nodiscard]] size_t getNumOwnedDofs() const { return static_cast< size_t >(m_num_owned); }
    [[nodiscard]] size_t getNumSharedDofs() const { return static_cast< size_t >(m_num_total - m_num_owned); }
    [[nodiscard]] size_t getNumTotalDofs() const { return static_cast< size_t >(m_num_total); }

    [[nodiscard]] auto size() const -> size_t { return m_map.size(); }
    [[nodiscard]] auto begin() const { return m_map.cbegin(); }
    [[nodiscard]] auto end() const { return m_map.cend(); }

private:
    util::ArrayOwner< std::array< dof_t, max_dofs_per_node > > m_map;
    dof_t                                                      m_num_owned{}, m_num_total{};
};

namespace detail
{
template < typename T >
inline constexpr bool is_node_map = false;
template < size_t dpn >
inline constexpr bool is_node_map< NodeToGlobalDofMap< dpn > > = true;
template < size_t dpn, size_t nm >
inline constexpr bool is_node_map< NodeToLocalDofMap< dpn, nm > > = true;
} // namespace detail

template < typename T >
concept NodeToDofMap_c = detail::is_node_map< T >;

template < size_t dofs_per_node >
template < CondensationPolicy CP >
NodeToGlobalDofMap< dofs_per_node >::NodeToGlobalDofMap(const node_interval_vector_t< dofs_per_node >& dof_intervals,
                                                        const NodeCondensationMap< CP >&               cond_map)
{
    L3STER_PROFILE_FUNCTION;
    if (cond_map.getCondensedIds().empty())
        return;
    if (not tryInitAsContiguous(dof_intervals, cond_map))
        initNonContiguous(dof_intervals, cond_map);
}

template < size_t dofs_per_node >
auto NodeToGlobalDofMap< dofs_per_node >::operator()(n_id_t node) const -> payload_t
{
    const auto map_ptr = std::get_if< map_t >(std::addressof(m_data));
    if (map_ptr)
        return map_ptr->at(node);
    else
    {
        const auto& contig_info = *std::get_if< ContiguousCaseInfo >(std::addressof(m_data));
        const auto  dof_inds    = std::span{contig_info.dof_inds.begin(), contig_info.n_dofs};
        auto        node_dof    = contig_info.base_dof + (node - contig_info.base_node) * contig_info.n_dofs;
        payload_t   retval;
        retval.fill(invalid_dof);
        for (auto i : dof_inds)
            retval[i] = node_dof++;
        return retval;
    }
}

template < size_t dofs_per_node >
template < CondensationPolicy CP >
bool NodeToGlobalDofMap< dofs_per_node >::tryInitAsContiguous(
    const node_interval_vector_t< dofs_per_node >& dof_intervals, const NodeCondensationMap< CP >& cond_map)
{
    const auto min_node_in_partition = cond_map.getCondensedIds().front();
    const auto max_node_in_partition = cond_map.getCondensedIds().back();
    auto       base_dof              = global_dof_t{};
    for (const auto& interval : dof_intervals)
    {
        const auto& [delim, coverage]    = interval;
        const auto [int_first, int_last] = delim;
        if (min_node_in_partition >= int_first and max_node_in_partition <= int_last)
        {
            auto dof_inds = std::array< std::uint8_t, dofs_per_node >{};
            auto write_it = begin(dof_inds);
            for (size_t i = 0; i < dofs_per_node; ++i)
                if (coverage.test(i))
                    *write_it++ = static_cast< std::uint8_t >(i);
            m_data = ContiguousCaseInfo{int_first, base_dof, dof_inds, static_cast< std::uint8_t >(coverage.count())};
            return true;
        }
        base_dof += (int_last - int_first + 1) * coverage.count();
    }
    return false;
}

template < size_t dofs_per_node >
template < CondensationPolicy CP >
void NodeToGlobalDofMap< dofs_per_node >::initNonContiguous(
    const node_interval_vector_t< dofs_per_node >& dof_intervals, const NodeCondensationMap< CP >& cond_map)
{
    auto&      map                 = m_data.template emplace< map_t >();
    const auto dof_interval_starts = computeIntervalStarts(dof_intervals);
    const auto compute_node_dofs   = [&](n_id_t node_id, ptrdiff_t interval_ind) {
        const auto dof_int_start = dof_interval_starts[interval_ind];
        const auto& [delim, cov] = dof_intervals[interval_ind];
        const auto [lo, hi]      = delim;
        auto retval              = std::array< global_dof_t, dofs_per_node >{};
        retval.fill(invalid_dof);
        global_dof_t node_dof = dof_int_start + (node_id - lo) * cov.count();
        for (ptrdiff_t i = 0; auto& dof : retval)
            if (cov.test(i++))
                dof = node_dof++;
        return retval;
    };
    for (auto search_it = begin(dof_intervals); auto n : cond_map.getCondensedIds())
    {
        search_it               = findNodeInterval(search_it, end(dof_intervals), n);
        const auto interval_ind = std::distance(begin(dof_intervals), search_it);
        const auto node_dofs    = compute_node_dofs(n, interval_ind);
        map.emplace(n, node_dofs);
    }
}

template < size_t dofs_per_node, size_t num_maps >
template < CondensationPolicy CP >
NodeToLocalDofMap< dofs_per_node, num_maps >::NodeToLocalDofMap(
    const NodeCondensationMap< CP >&           cond_map,
    const NodeToGlobalDofMap< dofs_per_node >& global_map,
    const std::same_as< tpetra_map_t > auto&... local_global_maps)
    requires(sizeof...(local_global_maps) == num_maps)
    : m_map(cond_map.getCondensedIds().size())
{
    L3STER_PROFILE_FUNCTION;
    const auto get_node_dofs = [&](n_id_t node, const tpetra_map_t& map) {
        auto retval = std::array< local_dof_t, dofs_per_node >{};
        std::ranges::transform(global_map(node), retval.begin(), [&](global_dof_t dof) {
            return NodeToGlobalDofMap< dofs_per_node >::isValid(dof) ? map.getLocalElement(dof) : invalid_dof;
        });
        return retval;
    };
    for (n_id_t cond_node : cond_map.getCondensedIds())
        m_map[cond_map.getUncondensedId(cond_node)] = payload_t{get_node_dofs(cond_node, local_global_maps)...};
}

template < size_t max_dofs_per_node >
LocalDofMap< max_dofs_per_node >::LocalDofMap(const NodeCondensationMap< CondensationPolicy::None >& cond_map,
                                              const NodeToGlobalDofMap< max_dofs_per_node >&         global_map,
                                              const mesh::NodeMap&                                   node_map,
                                              std::span< const global_dof_t >                        all_dofs,
                                              size_t                                                 num_owned_dofs)
    : m_map(cond_map.getCondensedIds().size()),
      m_num_owned{static_cast< dof_t >(num_owned_dofs)},
      m_num_total{static_cast< dof_t >(all_dofs.size())}
{
    L3STER_PROFILE_FUNCTION;
    auto       g2l           = util::IndexMap< global_dof_t, local_dof_t >{all_dofs};
    const auto translate_dof = [&g2l](global_dof_t gid) {
        return NodeToGlobalDofMap< max_dofs_per_node >::isValid(gid) ? g2l(gid) : invalid_dof;
    };
    for (auto cond_node : cond_map.getCondensedIds())
    {
        const auto& src_gids    = global_map(cond_node);
        const auto  uncond_node = cond_map.getUncondensedId(cond_node);
        const auto  local_node  = node_map.toLocal(uncond_node);
        auto&       dest_lids   = m_map.at(local_node);
        std::ranges::transform(src_gids, dest_lids.begin(), translate_dof);
    }
}
} // namespace lstr::dofs
#endif // L3STER_DOFS_NODETODOFMAP_HPP
