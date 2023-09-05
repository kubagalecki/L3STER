#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/util/Ranges.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <utility>

namespace lstr::mesh
{
namespace detail
{
template < el_o_t... orders >
using boundary_element_view_variant_t =
    parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView, orders... >;
template < el_o_t... orders >
using boundary_element_view_variant_array_t = util::ArrayOwner< boundary_element_view_variant_t< orders... > >;

template < el_o_t... orders >
auto makeBoundaryElementViews(const MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& bnd_ids)
    -> boundary_element_view_variant_array_t< orders... >
{
    const auto insert_pos_map  = std::invoke([&] {
        auto   retval = robin_hood::unordered_flat_map< el_id_t, size_t >{};
        size_t i      = 0;
        mesh.visit([&](const auto& element) { retval[element.getId()] = i++; }, bnd_ids);
        return retval;
    });
    const auto domain_dim_maps = makeDimToDomainMap(mesh);
    auto       retval          = boundary_element_view_variant_array_t< orders... >(insert_pos_map.size());
    auto       error_flag      = std::atomic_bool{false};
    const auto put_bnd_el_view = [&]< ElementType BET, el_o_t BEO >(const Element< BET, BEO >& bnd_el) {
        const auto bnd_el_nodes_sorted = util::getSortedArray(bnd_el.getNodes());
        const auto match_dom_el        = [&]< ElementType DET, el_o_t DEO >(const Element< DET, DEO >& dom_el) {
            const auto matched_side_opt = matchBoundaryNodesToElement(dom_el, bnd_el_nodes_sorted);
            if (matched_side_opt)
            {
                const auto el_boundary_view = BoundaryElementView{&dom_el, *matched_side_opt};
                const auto insert_pos       = insert_pos_map.at(bnd_el.getId());
                retval[insert_pos]          = el_boundary_view;
            }
            return matched_side_opt.has_value();
        };
        constexpr auto bnd_dim    = ElementTraits< Element< BET, BEO > >::native_dim;
        const auto&    domain_ids = domain_dim_maps.at(bnd_dim + 1);
        const auto     matched    = mesh.find(match_dom_el, domain_ids, std::execution::par);
        if (not matched)
            error_flag.store(true);
    };
    mesh.visit(put_bnd_el_view, bnd_ids, std::execution::par);
    util::throwingAssert(
        not error_flag.load(),
        "BoundaryView could not be constructed because some of the boundary elements are not edges/faces of "
        "any of the domain elements in the partition. This may be because the mesh was partitioned with "
        "incorrectly specified boundaries, resulting in the edge/face element being in a different partition "
        "from its parent area/volume element.");
    return retval;
}
} // namespace detail

template < typename Visitor, el_o_t... orders >
concept BoundaryViewVisitor_c = ElementDeductionHelper< orders... >::template invocable_on_boundary_views< Visitor >;

template < typename Zero, typename Transform, typename Reduction, el_o_t... orders >
concept BoundaryTransformReducible_c =
    ElementDeductionHelper< orders... >::template invocable_on_boundary_views_return< Zero, Transform > and
    ReductionFor_c< Reduction, Zero >;

template < el_o_t... orders >
class BoundaryView
{
    static_assert(sizeof...(orders) > 0);

public:
    BoundaryView(const MeshPartition< orders... >& mesh, util::ArrayOwner< d_id_t > bnd_ids)
        : m_boundary_elements{detail::makeBoundaryElementViews(mesh, bnd_ids)},
          m_parent_partition{&mesh},
          m_boundary_ids(std::move(bnd_ids))
    {}

    template < BoundaryViewVisitor_c< orders... > Visitor,
               ExecutionPolicy_c                  ExecPolicy = MeshPartition< orders... >::DefaultExec >
    void visit(Visitor&& visitor, ExecPolicy&& policy = {}) const;
    template < std::copy_constructible Zero,
               std::copy_constructible Transform,
               std::copy_constructible Reduction,
               ExecutionPolicy_c       ExecPolicy = std::execution::sequenced_policy >
    auto reduce(Zero zero, Transform trans, Reduction reduction, ExecPolicy policy = {}) const -> Zero
        requires BoundaryTransformReducible_c< Zero, Transform, Reduction, orders... >;

    [[nodiscard]] auto getParent() const -> const MeshPartition< orders... >* { return m_parent_partition; }
    [[nodiscard]] auto size() const -> size_t { return m_boundary_elements.size(); }
    [[nodiscard]] auto getIds() const -> std::span< const d_id_t > { return m_boundary_ids; }

private:
    detail::boundary_element_view_variant_array_t< orders... > m_boundary_elements;
    const MeshPartition< orders... >*                          m_parent_partition{};
    util::ArrayOwner< d_id_t >                                 m_boundary_ids;
};

template < el_o_t... orders >
template < BoundaryViewVisitor_c< orders... > Visitor, ExecutionPolicy_c ExecPolicy >
void BoundaryView< orders... >::visit(Visitor&& visitor, ExecPolicy&& policy) const
{
    if constexpr (std::same_as< std::remove_cvref_t< ExecPolicy >, std::execution::sequenced_policy >)
        std::ranges::for_each(m_boundary_elements,
                              [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
    else
        std::for_each(std::forward< ExecPolicy >(policy),
                      m_boundary_elements.begin(),
                      m_boundary_elements.end(),
                      [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
}

template < el_o_t... orders >
template < std::copy_constructible Zero,
           std::copy_constructible Transform,
           std::copy_constructible Reduction,
           ExecutionPolicy_c       ExecPolicy >
auto BoundaryView< orders... >::reduce(Zero zero, Transform trans, Reduction reduction, ExecPolicy policy) const -> Zero
    requires BoundaryTransformReducible_c< Zero, Transform, Reduction, orders... >
{
    const auto transform_variant = [&](const auto& var) {
        return std::visit< Zero >(trans, var);
    };
    return std::transform_reduce(
        policy, m_boundary_elements.begin(), m_boundary_elements.end(), zero, reduction, transform_variant);
}

template < el_o_t... orders >
class BoundaryViewManager
{
public:
    BoundaryViewManager() = default;
    inline BoundaryViewManager(const MeshPartition< orders... >& mesh, const util::ArrayOwner< d_id_t >& bnd_ids);

    auto get(d_id_t id) const -> const BoundaryView< orders... >& { return m_boundary_views.at(id); }

private:
    std::map< d_id_t, BondaryView< orders... > > m_boundary_views;
};

template < el_o_t... orders >
BoundaryViewManager< orders... >::BoundaryViewManager(const MeshPartition< orders... >& mesh,
                                                      const util::ArrayOwner< d_id_t >& bnd_ids)
{
    for (d_id_t id : bnd_ids)
        m_boundary_views[id] = BoundaryView{mesh, std::views::single(id)};
}
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
