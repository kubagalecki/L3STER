#include "mesh/ElementInitializer.hpp"

#ifndef L3STER_MACROS_MESH_INIT_ELEMENT
#define L3STER_MACROS_MESH_INIT_ELEMENT

// Macro which takes care of namespaces for convenience
#define INIT_ELEMENT(ELTYPE, ELORDER) \
lstr::mesh::initializeElement(lstr::mesh::ElementInitializer< \
lstr::mesh::ElementTypes::ELTYPE, ELORDER >{})

void lstr::mesh::init_elements()
{
    INIT_ELEMENT(Quad, 2);
    INIT_ELEMENT(Quad, 3);
}
#endif
