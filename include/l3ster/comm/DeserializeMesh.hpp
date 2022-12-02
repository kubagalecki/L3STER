#ifndef L3STER_COMM_DESERIALIZEMESH_HPP
#define L3STER_COMM_DESERIALIZEMESH_HPP

#include "SerializeMesh.hpp"
#include "l3ster/mesh/ElementTraits.hpp"

namespace lstr
{
namespace detail
{
template < ElementTypes T, el_o_t O >
std::vector< Element< T, O > >
deserializeElementVector(const n_id_t*& nodes, const val_t*& data, const el_id_t*& ids, size_t size)
{
    using node_array_t = Element< T, O >::node_array_t;
    using data_array_t = Element< T, O >::element_data_t;

    constexpr auto node_chunk_size = Element< T, O >::n_nodes;

    std::vector< Element< T, O > > retval;
    retval.reserve(size);

    for (size_t el_i = 0; el_i < size; ++el_i)
    {
        node_array_t node_array;
        std::copy_n(nodes, node_chunk_size, begin(node_array));
        std::advance(nodes, node_chunk_size);

        data_array_t data_array;
        for (ptrdiff_t vert_i = 0; vert_i < static_cast< ptrdiff_t >(ElementData< T, O >::n_verts); ++vert_i)
            for (ptrdiff_t dim_i = 0; dim_i < 3; ++dim_i)
                data_array.vertices[vert_i][dim_i] = *data++; // NOLINT

        const auto el_id = *ids++; // NOLINT

        retval.emplace_back(node_array, data_array, el_id);
    }

    return retval;
}

inline auto initElementVectorVariant(ElementTypes T, el_o_t O)
{
    using ret_t = Domain::element_vector_variant_t;
    ret_t      retval;
    const auto init_if = [&]< ElementTypes TYPE, el_o_t ORDER >(ValuePack< TYPE, ORDER >) {
        if (T == TYPE and O == ORDER)
        {
            retval.template emplace< std::vector< Element< TYPE, ORDER > > >();
            return true;
        }
        else
            return false;
    };

    const auto deduce = [&]< typename... Types >(TypePack< Types... >) {
        (init_if(Types{}) or ...);
    };
    deduce(type_order_combinations{});
    return retval;
}

inline Domain deserializeDomain(const SerializedDomain& domain)
{
    Domain::element_vector_variant_vector_t elements;

    auto node_ptr = domain.element_nodes.data();
    auto data_ptr = domain.element_data.data();
    auto id_ptr   = domain.element_ids.data();

    size_t i = 0;
    for (auto&& offset : domain.type_order_offsets)
    {
        auto& current_vec = elements.emplace_back(
            initElementVectorVariant(static_cast< ElementTypes >(domain.types[i]), domain.orders[i]));
        std::visit(
            [&]< ElementTypes T, el_o_t O >(std::vector< Element< T, O > >& v) {
                v = deserializeElementVector< T, O >(node_ptr, data_ptr, id_ptr, offset);
            },
            current_vec);
        ++i;
    }

    dim_t domain_dim = 0;
    if (i > 0u)
        std::visit([&]< ElementTypes T, el_o_t O >(
                       const std::vector< Element< T, O > >&) { domain_dim = Element< T, O >::native_dim; },
                   elements[0]);

    return Domain{std::move(elements), domain_dim};
}
} // namespace detail

inline MeshPartition deserializePartition(const SerializedPartition& partition)
{
    MeshPartition::domain_map_t domain_map;
    for (const auto& [id, domain] : partition.m_domains)
        domain_map.emplace(id, detail::deserializeDomain(domain));
    return MeshPartition{std::move(domain_map), partition.m_nodes, partition.m_n_owned_nodes};
}

inline MeshPartition deserializePartition(SerializedPartition&& partition)
{
    MeshPartition::domain_map_t domain_map;
    for (const auto& [id, domain] : partition.m_domains)
        domain_map.emplace(id, detail::deserializeDomain(domain));
    return MeshPartition{std::move(domain_map), std::move(partition.m_nodes), partition.m_n_owned_nodes};
}
} // namespace lstr
#endif // L3STER_COMM_DESERIALIZEMESH_HPP
