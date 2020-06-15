// Data structure representing a mesh node

#ifndef L3STER_INCGUARD_MESH_NODE_HPP
#define L3STER_INCGUARD_MESH_NODE_HPP

#include "typedefs/Types.h"

#include <array>

namespace lstr
{
namespace mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                       NODE CLASS                                         //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Node class - essentially a container for node coordinates, templated with space dimension
*/
template < types::dim_t DIM >
class Node
{
public:
    using array_t = std::array< types::val_t, DIM >;

    // Ctors & Dtors
    Node() = delete;

    Node(const array_t& _coords) : coords(_coords) {}

    Node(const Node&) = default;

    Node(Node&&) noexcept = default;

    Node& operator=(const Node&) = default;

    Node& operator=(Node&&) noexcept = default;

    ~Node() = default;

    // access
    const array_t& getCoords() { return coords; }

    void setCoords(const array_t& _coords) { coords = _coords; }

private:
    array_t coords;
};
} // namespace mesh
} // namespace lstr

#endif // end include guard
