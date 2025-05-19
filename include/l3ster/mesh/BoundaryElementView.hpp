#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementMeta.hpp"
#include "l3ster/util/Ranges.hpp"

namespace lstr::mesh
{
template < ElementType ET, el_o_t EO >
class BoundaryElementView
{
public:
    static constexpr auto type  = ET;
    static constexpr auto order = EO;

    BoundaryElementView() = default;
    BoundaryElementView(const Element< ET, EO >* parent, const el_side_t side) : m_parent{parent}, m_side{side} {}

    [[nodiscard]] const Element< ET, EO >* operator->() const { return m_parent; }
    [[nodiscard]] const Element< ET, EO >& operator*() const { return *m_parent; }
    [[nodiscard]] auto                     getSide() const { return m_side; }
    [[nodiscard]] inline auto              getSideNodeInds() const -> std::span< const el_locind_t >;
    [[nodiscard]] inline auto              getSideNodesView() const;

private:
    const Element< ET, EO >* m_parent{};
    el_side_t                m_side{};
};

template < ElementType ET, el_o_t EO >
auto BoundaryElementView< ET, EO >::getSideNodesView() const
{
    return util::makeIndexedView(std::span{m_parent->nodes}, getSideNodeInds());
}

template < ElementType ET, el_o_t EO >
auto BoundaryElementView< ET, EO >::getSideNodeInds() const -> std::span< const el_locind_t >
{
    return getSideNodeIndices< ET, EO >(m_side);
}
} // namespace lstr::mesh
#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
