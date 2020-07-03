// Data type representing a physical element

#ifndef L3STER_INCGUARD_MESH_ELEMENT_HPP
#define L3STER_INCGUARD_MESH_ELEMENT_HPP

#include <algorithm>
#include <array>
#include <cstddef>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace lstr::mesh
{
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class Element
{
public:
    using node_array_t =
        std::array< types::n_id_t, ElementTraits< Element< ELTYPE, ELORDER > >::nodes_per_element >;
    using node_array_ref_t      = node_array_t&;
    using node_array_constref_t = const node_array_t&;

    Element()                   = delete;
    Element(const Element&)     = default;
    Element(Element&&) noexcept = default;
    Element& operator=(const Element&) = default;
    Element& operator=(Element&&) noexcept = default;

    // Element is constructible from any unsigned array type
    template <
        typename UINT,
        std::enable_if_t< std::is_unsigned_v< UINT > && !std::is_same_v< UINT, types::n_id_t >,
                          bool > = true >
    explicit Element(const std::array< UINT, std::tuple_size_v< node_array_t > >& nodes_);
    explicit Element(const node_array_t& nodes_) : nodes{nodes_} {}

    [[nodiscard]] const node_array_t& getNodes() const noexcept { return nodes; }
    [[nodiscard]] node_array_t&       getNodesRef() noexcept { return nodes; }

private:
    node_array_t                                                      nodes;
    typename ElementTraits< Element< ELTYPE, ELORDER > >::ElementData data{};
};

template < ElementTypes ELTYPE, types::el_o_t ELORDER >
template <
    typename UINT,
    std::enable_if_t< std::is_unsigned_v< UINT > && !std::is_same_v< UINT, types::n_id_t >, bool > >
Element< ELTYPE, ELORDER >::Element(
    const std::array< UINT, std::tuple_size_v< node_array_t > >& nodes_)
{
    std::transform(nodes_.cbegin(), nodes.cend(), nodes.begin(), [](const UINT& node) {
        return static_cast< types::n_id_t >(node);
    });
}
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_ELEMENT_HPP
