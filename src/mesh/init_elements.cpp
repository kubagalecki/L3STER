// Macro which takes care of namespaces for convenience
#include "mesh/ElementInitializer.hpp"
#include "mesh/ElementTypes.hpp"
#include "typedefs/Types.h"

#define INIT_ELEMENT(ELTYPE, ELORDER)                                                              \
    initializeElement(ElementInitializer< ElementTypes::ELTYPE, ELORDER >{})

namespace lstr
{
namespace mesh
{
void init_elements()
{
    INIT_ELEMENT(Quad, 2);
    INIT_ELEMENT(Quad, 3);
}
} // namespace mesh
} // namespace lstr
