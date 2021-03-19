// Data structure representing a mesh node

#ifndef L3STER_INCGUARD_MESH_NODE_HPP
#define L3STER_INCGUARD_MESH_NODE_HPP

#include "defs/Typedefs.h"

#include <array>

namespace lstr
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                       NODE CLASS                                         //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Node class - aggregate of node dimensions
*/
template < dim_t DIM >
struct Node
{
    std::array< val_t, DIM > coords;
};
} // namespace lstr

#endif // end include guard
