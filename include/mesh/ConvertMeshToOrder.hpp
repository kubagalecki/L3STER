#ifndef L3STER_MESH_CONVERTMESHTOORDER_HPP
#define L3STER_MESH_CONVERTMESHTOORDER_HPP
#include "mesh/ElementIntersecting.hpp"
#include "mesh/Mesh.hpp"
#include "util/Common.hpp"
#include "util/MetisUtils.hpp"

namespace lstr
{
template < el_o_t ORDER >
void convertMeshToOrder(Mesh& mesh)
{
    if (mesh.getPartitions().size() != 1)
        throw std::logic_error{"Cannot convert a mesh which is either empty or has been partitioned"};

    auto&       part      = mesh.getPartitions()[0];
    const auto& dual_mesh = part.initDualGraph();

    n_id_t new_node = part.getNodes().size();
}
} // namespace lstr
#endif // L3STER_MESH_CONVERTMESHTOORDER_HPP
