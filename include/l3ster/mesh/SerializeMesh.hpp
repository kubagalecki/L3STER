#ifndef L3STER_MESH_SERIALIZEMESH_HPP
#define L3STER_MESH_SERIALIZEMESH_HPP

#include "l3ster/mesh/Domain.hpp"
#include "l3ster/mesh/MeshPartition.hpp"

#include <iterator>

namespace lstr::mesh
{
namespace detail
{
template < mesh::ElementType T, el_o_t O >
void serializeElementNodes(const mesh::Element< T, O >& el, n_id_t* dest) noexcept
{
    std::ranges::copy(el.getNodes(), dest);
}

template < mesh::ElementType T, el_o_t O >
void serializeElementData(const mesh::Element< T, O >& el, val_t* dest) noexcept
{
    constexpr auto n_verts = static_cast< ptrdiff_t >(mesh::ElementData< T, O >::n_verts);
    auto&          verts   = el.getData().vertices;
    for (ptrdiff_t vert_i = 0; vert_i < n_verts; ++vert_i)
        for (ptrdiff_t dim_i = 0; dim_i < 3; ++dim_i)
            dest[vert_i * 3 + dim_i] = verts[vert_i][dim_i];
}

class ElementSerializer
{
public:
    ElementSerializer(n_id_t* node_init_offset, val_t* data_init_offset, el_id_t* id_init_offset)
        : node_offset(node_init_offset), data_offset(data_init_offset), id_offset(id_init_offset)
    {}

    template < mesh::ElementType T, el_o_t O >
    void operator()(const mesh::Element< T, O >& el) noexcept
    {
        serializeElementNodes(el, node_offset);
        std::advance(node_offset, mesh::Element< T, O >::n_nodes);

        serializeElementData(el, data_offset);
        std::advance(data_offset, mesh::ElementData< T, O >::n_verts * 3);

        *id_offset = el.getId();
        std::advance(id_offset, 1);
    }

private:
    n_id_t*  node_offset;
    val_t*   data_offset;
    el_id_t* id_offset;
};
} // namespace detail

struct SerializedDomain
{
    SerializedDomain() = default;
    template < el_o_t... orders >
    explicit SerializedDomain(const mesh::Domain< orders... >& domain);

    std::vector< n_id_t >            element_nodes;
    std::vector< val_t >             element_data;
    std::vector< el_id_t >           element_ids;
    std::vector< ptrdiff_t >         type_order_offsets;
    std::vector< std::uint_fast8_t > types;
    std::vector< el_o_t >            orders;
};

template < el_o_t... el_orders >
SerializedDomain::SerializedDomain(const mesh::Domain< el_orders... >& domain)
{
    size_t       node_size{0u}, data_size{0u}, id_size{0u};
    const size_t offset_size{domain.m_element_vectors.size()};
    type_order_offsets.reserve(offset_size);
    types.reserve(offset_size);
    orders.reserve(offset_size);

    const auto update_sizes = [&]< mesh::ElementType T, el_o_t O >(const std::vector< mesh::Element< T, O > >& el_v) {
        constexpr auto node_chunk_size = mesh::Element< T, O >::n_nodes;
        constexpr auto data_chunk_size = mesh::ElementData< T, O >::n_verts * 3;
        const auto     vec_size        = el_v.size();

        node_size += vec_size * node_chunk_size;
        data_size += vec_size * data_chunk_size;
        id_size += vec_size;

        type_order_offsets.push_back(vec_size);
        types.push_back(static_cast< std::uint_fast8_t >(T));
        orders.push_back(O);
    };

    for (const auto& el_v : domain.m_element_vectors)
        std::visit(update_sizes, el_v);

    element_nodes.resize(node_size);
    element_data.resize(data_size);
    element_ids.resize(id_size);

    detail::ElementSerializer serializer{element_nodes.data(), element_data.data(), element_ids.data()};
    domain.visit(serializer, std::execution::seq);
}

struct SerializedPartition
{
    SerializedPartition() = default;
    template < el_o_t... orders >
    explicit SerializedPartition(const mesh::MeshPartition< orders... >& part)
        : nodes{copy(part.m_nodes)}, n_owned_nodes{part.m_n_owned_nodes}, boundaries{part.getBoundaryIdsCopy()}
    {
        for (const auto& [id, dom] : part.m_domains)
            domains.emplace(id, dom);
    }
    template < el_o_t... orders >
    explicit SerializedPartition(mesh::MeshPartition< orders... >&& part)
        : nodes{std::move(part.m_nodes)}, n_owned_nodes{part.m_n_owned_nodes}, boundaries{part.getBoundaryIdsCopy()}
    {
        for (const auto& [id, dom] : part.m_domains)
            domains.emplace(id, dom);
    }

    std::map< d_id_t, SerializedDomain > domains;
    util::ArrayOwner< n_id_t >           nodes;
    size_t                               n_owned_nodes{};
    util::ArrayOwner< d_id_t >           boundaries;
};
} // namespace lstr::mesh
#endif // L3STER_MESH_SERIALIZEMESH_HPP