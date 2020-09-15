// Data structure representing a mesh node

#ifndef L3STER_INCGUARD_MESH_NODE_HPP
#define L3STER_INCGUARD_MESH_NODE_HPP

#include <array>

namespace lstr::mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                       NODE CLASS                                         //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Node class - aggregate of node dimensions
*/
template < types::dim_t DIM >
struct Node
{
    std::array< types::val_t, DIM > coords;
};
} // namespace lstr::mesh

#endif // end include guard
