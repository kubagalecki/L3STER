#ifndef L3STER_INCGUARD_MESH_MESH_HPP
#define L3STER_INCGUARD_MESH_MESH_HPP

#include "definitions/Constants.hpp"
#include "definitions/Typedefs.h"
#include "mesh/MeshPartition.hpp"
#include "mesh/Node.hpp"

#include <variant>
#include <vector>

namespace lstr::mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                         MESH CLASS                                       //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Mesh - top level interface
*/
class Mesh
{
public:
    template < types::dim_t DIM >
    using node_vector_t         = std::vector< Node< DIM > >;
    using node_vector_variant_t = parametrize_over_dims_t< std::variant, node_vector_t >;

private:
    node_vector_variant_t        nodes;
    std::vector< MeshPartition > partitions;
};
} // namespace lstr::mesh

#endif // L3STER_INCGUARD_MESH_MESH_HPP
