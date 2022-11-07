#ifndef L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
#define L3STER_MESH_BOUNDARYELEMENTVIEW_HPP

#include "l3ster/mesh/Aliases.hpp"
#include "l3ster/mesh/Element.hpp"

namespace lstr
{
template < ElementTypes ET, el_o_t EO >
class BoundaryElementView
{
public:
    BoundaryElementView(const Element< ET, EO >& element, const el_side_t side)
        : m_element_ptr{std::addressof(element)}, m_element_side{side}
    {}

    [[nodiscard]] const Element< ET, EO >* operator->() const { return m_element_ptr; }
    [[nodiscard]] const Element< ET, EO >& operator*() const { return *m_element_ptr; }
    [[nodiscard]] auto                     getSide() const { return m_element_side; }

private:
    const Element< ET, EO >* m_element_ptr;
    el_side_t                m_element_side;
};
} // namespace lstr
#endif // L3STER_MESH_BOUNDARYELEMENTVIEW_HPP
