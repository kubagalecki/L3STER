// Top level data structure for storing the entire mesh

#ifndef L3STER_INCGUARD_MESH_MESHMASTER_HPP
#define L3STER_INCGUARD_MESH_MESHMASTER_HPP

#include "mesh/Domain.hpp"
#include "mesh/Node.hpp"
#include "typedefs/Types.h"
//#include "mesh/ElementInitializer.hpp"

#include <string_view>
#include <vector>

namespace lstr::mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                     MESH MASTER CLASS                                    //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Mesh master class - highest level mesh interface. Templated with space dimension
*/
template < types::dim_t DIM >
class MeshMaster
{
public:
    // Ctor & Dtors
    MeshMaster() = default;
    MeshMaster(const MeshMaster&) = default;
    MeshMaster(MeshMaster&&) noexcept = default;
    MeshMaster& operator=(const MeshMaster&) = default;
    MeshMaster& operator=(MeshMaster&&) noexcept = default;

    ~MeshMaster() = default;

    void meshRead(const std::string_view&);

    void meshAppend(const std::string_view&);

    void meshWrite(const std::string_view&);

private:
    std::map< types::d_id_t, Domain > elems;
    std::vector< Node< DIM > >        nodes;
};
} // namespace lstr::mesh

#endif // end include guard
