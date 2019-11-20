// Top level data structure for storing the entire mesh

#ifndef L3STER_INCGUARD_MESH_MESHMASTER_HPP
#define L3STER_INCGUARD_MESH_MESHMASTER_HPP

#include "mesh/Domain.hpp"
#include "mesh/Node.hpp"
#include "typedefs/Types.h"

#include <string>
#include <vector>

namespace lstr
{
    namespace mesh
    {
        //////////////////////////////////////////////////////////////////////////////////////////////
        //                                     MESH MASTER CLASS                                    //
        //////////////////////////////////////////////////////////////////////////////////////////////
        /*
        Mesh master class - highest level mesh interface. Templated with space dimension
        */
        template <types::dim_t DIM>
        class MeshMaster
        {
            // Ctor & Dtors
            MeshMaster()    = default;
            ~MeshMaster()   = default;
            
            void meshRead(std::string);
            void meshAppend(std::string);
            void meshWrite(std::string);
        private:
            std::map<types::d_id_t, Domain>     elems;
            std::vector< Node<DIM> >            nodes;
        };
    }
}

#endif      // end include guard
