#ifndef L3STER_MESH_BOUNDARYVIEW_HPP
#define L3STER_MESH_BOUNDARYVIEW_HPP

#include "l3ster/mesh/BoundaryElementView.hpp"
#include "l3ster/mesh/Domain.hpp"
#include "l3ster/mesh/MeshUtils.hpp"
#include "l3ster/util/Ranges.hpp"
#include "l3ster/util/RobinHoodHashTables.hpp"

#include <utility>

namespace lstr::mesh
{
template < el_o_t... orders >
class BoundaryView
{
    static_assert(sizeof...(orders) > 0);

    using Constraint = ElementDeductionHelper< orders... >;

public:
    using boundary_element_view_variant_t =
        parametrize_type_over_element_types_and_orders_t< std::variant, BoundaryElementView, orders... >;
    using boundary_element_view_variant_vector_t = std::vector< boundary_element_view_variant_t >;

    template < RangeOfConvertibleTo_c< d_id_t > Ids >
    BoundaryView(const MeshPartition< orders... >& mesh, Ids&& bnd_ids)
        : m_boundary_elements{makeBoundaryElementViews(mesh, bnd_ids)},
          m_parent_partition{&mesh},
          m_boundary_ids(util::toVector(util::castView< d_id_t >(std::forward< Ids >(bnd_ids))))
    {}

    template < typename F, ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    void visit(F&& visitor, ExecPolicy&& policy = {}) const
        requires(Constraint::template invocable_on_boundary_views< F >);
    template < typename Zero,
               typename Proj,
               typename Red,
               ExecutionPolicy_c ExecPolicy = std::execution::sequenced_policy >
    Zero reduce(Zero&& zero, Proj&& projection, Red&& reduction, ExecPolicy policy = {}) const
        requires(Constraint::template invocable_on_boundary_views_return< Zero, Proj > and
                 requires(std::remove_cvref_t< Zero > z, std::remove_cvref_t< Red > r) {
                     {
                         std::invoke(
                             std::forward< Red >(reduction), std::forward< Zero >(zero), std::forward< Zero >(zero))
                     } -> std::convertible_to< std::remove_cvref_t< Zero > >;
                 });

    [[nodiscard]] auto getParent() const -> const MeshPartition< orders... >* { return m_parent_partition; }
    [[nodiscard]] auto size() const -> size_t { return m_boundary_elements.size(); }
    [[nodiscard]] auto getIds() const -> std::span< const d_id_t > { return m_boundary_ids; }

private:
    template < RangeOfConvertibleTo_c< d_id_t > Ids >
    static auto makeBoundaryElementViews(const MeshPartition< orders... >& mesh, Ids&& ids)
        -> boundary_element_view_variant_vector_t;

    boundary_element_view_variant_vector_t m_boundary_elements;
    const MeshPartition< orders... >*      m_parent_partition{};
    std::vector< d_id_t >                  m_boundary_ids;
};

template < el_o_t... orders >
template < RangeOfConvertibleTo_c< d_id_t > Ids >
auto BoundaryView< orders... >::makeBoundaryElementViews(const MeshPartition< orders... >& mesh, Ids&& bnd_ids)
    -> boundary_element_view_variant_vector_t
{
    const auto insert_pos_map  = std::invoke([&] {
        auto   retval = robin_hood::unordered_flat_map< el_id_t, size_t >{};
        size_t i      = 0;
        mesh.visit([&](const auto& element) { retval[element.getId()] = i++; }, bnd_ids);
        return retval;
    });
    const auto domain_dim_maps = makeDimToDomainMap(mesh);
    auto       retval          = boundary_element_view_variant_vector_t(insert_pos_map.size());
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
    mesh.visit(put_bnd_el_view, std::forward< Ids >(bnd_ids), std::execution::par);
    util::throwingAssert(
        not error_flag.load(),
        "BoundaryView could not be constructed because some of the boundary elements are not edges/faces of "
        "any of the domain elements in the partition. This may be because the mesh was partitioned with "
        "incorrectly specified boundaries, resulting in the edge/face element being in a different partition "
        "from its parent area/volume element.");
    return retval;
}

template < el_o_t... orders >
template < typename F, ExecutionPolicy_c ExecPolicy >
void BoundaryView< orders... >::visit(F&& visitor, ExecPolicy&& policy) const
    requires(Constraint::template invocable_on_boundary_views< F >)
{
    if constexpr (std::same_as< std::remove_cvref_t< ExecPolicy >, std::execution::sequenced_policy >)
    {
        std::ranges::for_each(m_boundary_elements,
                              [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
    }
    else
        std::for_each(std::forward< ExecPolicy >(policy),
                      begin(m_boundary_elements),
                      end(m_boundary_elements),
                      [&visitor](const auto& boundary_el) { std::visit(visitor, boundary_el); });
}

template < el_o_t... orders >
template < typename Zero, typename Proj, typename Red, ExecutionPolicy_c ExecPolicy >
Zero BoundaryView< orders... >::reduce(Zero&& zero, Proj&& projection, Red&& reduction, ExecPolicy policy) const
    requires(Constraint::template invocable_on_boundary_views_return< Zero, Proj > and
             requires(std::remove_cvref_t< Zero > z, std::remove_cvref_t< Red > r) {
                 {
                     std::invoke(std::forward< Red >(reduction), std::forward< Zero >(zero), std::forward< Zero >(zero))
                 } -> std::convertible_to< std::remove_cvref_t< Zero > >;
             })
{
    return std::transform_reduce(
        policy,
        begin(m_boundary_elements),
        end(m_boundary_elements),
        std::forward< Zero >(zero),
        std::forward< Red >(reduction),
        [&](const boundary_element_view_variant_t& var) { return std::visit< Zero >(projection, var); });
}
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYVIEW_HPP
