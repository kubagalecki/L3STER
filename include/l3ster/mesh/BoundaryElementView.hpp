#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "l3ster/mesh/Element.hpp"
#include "l3ster/mesh/ElementMeta.hpp"

namespace lstr::mesh
{
template < ElementType ET, el_o_t EO >
class BoundaryElementView
{
public:
    BoundaryElementView(const Element< ET, EO >& element, const el_side_t side)
        : m_element_ptr{std::addressof(element)}, m_element_side{side}
    {}

    [[nodiscard]] const Element< ET, EO >* operator->() const { return m_element_ptr; }
    [[nodiscard]] const Element< ET, EO >& operator*() const { return *m_element_ptr; }
    [[nodiscard]] auto                     getSide() const { return m_element_side; }
    [[nodiscard]] inline auto              getSideNodeInds() const -> std::span< const el_locind_t >;

private:
    const Element< ET, EO >* m_element_ptr;
    el_side_t                m_element_side;
};

template < ElementType ET, el_o_t EO >
auto BoundaryElementView< ET, EO >::getSideNodeInds() const -> std::span< const el_locind_t >
{
    std::span< const el_locind_t > retval;
    const auto                     fold_expr_helper = [&]< el_side_t side >(std::integral_constant< el_side_t, side >) {
        if (side == getSide())
        {
            retval = std::get< side >(ElementTraits< Element< ET, EO > >::boundary_table);
            return true;
        }
        else
            return false;
    };
    std::invoke(
        [&]< el_side_t... sides >(std::integer_sequence< el_side_t, sides... >) {
            (fold_expr_helper(std::integral_constant< el_side_t, sides >{}) or ...);
        },
        std::make_integer_sequence< el_side_t, ElementTraits< Element< ET, EO > >::n_sides >{});
    return retval;
}
} // namespace lstr
#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
