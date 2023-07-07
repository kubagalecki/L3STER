#ifndef L3STER_COMM_DESERIALIZEMESH_HPP
#define L3STER_COMM_DESERIALIZEMESH_HPP

#include "SerializeMesh.hpp"
#include "l3ster/mesh/ElementTraits.hpp"

namespace lstr
{
namespace detail
{
template < mesh::ElementType T, el_o_t O >
auto deserializeElementVector(const n_id_t*& nodes, const val_t*& data, const el_id_t*& ids, size_t size)
    -> std::vector< mesh::Element< T, O > >
{
    using node_array_t             = mesh::Element< T, O >::node_array_t;
    using data_array_t             = mesh::Element< T, O >::element_data_t;
    constexpr auto node_chunk_size = mesh::Element< T, O >::n_nodes;
    auto           retval          = std::vector< mesh::Element< T, O > >{};
    retval.reserve(size);
    for (size_t el_i = 0; el_i < size; ++el_i)
    {
        node_array_t node_array;
        std::copy_n(nodes, node_chunk_size, begin(node_array));
        std::advance(nodes, node_chunk_size);
        data_array_t data_array;
        for (ptrdiff_t vert_i = 0; vert_i < static_cast< ptrdiff_t >(mesh::ElementData< T, O >::n_verts); ++vert_i)
            for (ptrdiff_t dim_i = 0; dim_i < 3; ++dim_i)
                data_array.vertices[vert_i][dim_i] = *data++;
        const auto el_id = *ids++;
        retval.emplace_back(node_array, data_array, el_id);
    }
    return retval;
}

template < el_o_t... orders >
auto initElementVectorVariant(mesh::ElementType T, el_o_t O) -> mesh::Domain< orders... >::element_vector_variant_t
{
    auto       retval  = typename mesh::Domain< orders... >::element_vector_variant_t{};
    const auto init_if = [&]< mesh::ElementType TYPE, el_o_t ORDER >(util::ValuePack< TYPE, ORDER >) {
        if (T == TYPE and O == ORDER)
        {
            retval.template emplace< std::vector< mesh::Element< TYPE, ORDER > > >();
            return true;
        }
        else
            return false;
    };

    const auto deduce = [&]< typename... Types >(util::TypePack< Types... >) {
        (init_if(Types{}) or ...);
    };
    deduce(mesh::type_order_combinations< orders... >{});
    return retval;
}

template < el_o_t... orders >
auto deserializeDomain(const SerializedDomain& domain) -> mesh::Domain< orders... >
{
    auto elements = typename mesh::Domain< orders... >::element_vector_variant_vector_t{};

    auto node_ptr = domain.element_nodes.data();
    auto data_ptr = domain.element_data.data();
    auto id_ptr   = domain.element_ids.data();

    size_t i = 0;
    for (auto&& offset : domain.type_order_offsets)
    {
        auto& current_vec = elements.emplace_back(
            initElementVectorVariant< orders... >(static_cast< mesh::ElementType >(domain.types[i]), domain.orders[i]));
        std::visit(
            [&]< mesh::ElementType T, el_o_t O >(std::vector< mesh::Element< T, O > >& v) {
                v = deserializeElementVector< T, O >(node_ptr, data_ptr, id_ptr, offset);
            },
            current_vec);
        ++i;
    }

    dim_t domain_dim = 0;
    if (i > 0u)
        std::visit([&]< mesh::ElementType T, el_o_t O >(
                       const std::vector< mesh::Element< T, O > >&) { domain_dim = mesh::Element< T, O >::native_dim; },
                   elements.front());

    return {std::move(elements), domain_dim};
}
} // namespace detail

template < el_o_t... orders >
auto deserializePartition(const SerializedPartition& partition) -> mesh::MeshPartition< orders... >
{
    auto domain_map = typename mesh::MeshPartition< orders... >::domain_map_t{};
    for (const auto& [id, domain] : partition.m_domains)
        domain_map.emplace(id, detail::deserializeDomain< orders... >(domain));
    return {std::move(domain_map), partition.m_nodes, partition.m_n_owned_nodes};
}

template < el_o_t... orders >
auto deserializePartition(SerializedPartition&& partition) -> mesh::MeshPartition< orders... >
{
    auto domain_map = typename mesh::MeshPartition< orders... >::domain_map_t{};
    for (const auto& [id, domain] : partition.m_domains)
        domain_map.emplace(id, detail::deserializeDomain< orders... >(domain));
    return {std::move(domain_map), std::move(partition.m_nodes), partition.m_n_owned_nodes};
}
} // namespace lstr
#endif // L3STER_COMM_DESERIALIZEMESH_HPP
