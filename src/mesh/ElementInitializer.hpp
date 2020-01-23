// Class for forcing the compilation of specific elements

#ifndef L3STER_INCGUARD_MESH_ELEMENTINITIALIZER_HPP
#define L3STER_INCGUARD_MESH_ELEMENTINITIALIZER_HPP

#include "mesh/ElementTypes.hpp"
#include "mesh/ElementVector.hpp"
#include "typedefs/Types.h"
#include "utility/Factory.hpp"

#include <vector>
#include <algorithm>
#include <utility>

namespace lstr
{
namespace mesh
{

void init_elements();

auto initializeElementInitializerMaster()
{
    init_elements();
    return true;
}

class ElementInitializerMaster
{
    template <typename T>
    friend void initializeElement(T);

private:
    static std::vector< std::pair<ElementTypes, types::el_o_t> > initialized_elements;
    static bool init_status;
};

std::vector< std::pair<ElementTypes, types::el_o_t> >
ElementInitializerMaster::initialized_elements{};

bool ElementInitializerMaster::init_status =
    initializeElementInitializerMaster();

template <ElementTypes ELTYPE, types::el_o_t ELORDER>
class ElementInitializer
{
public:
    static constexpr auto el_t  = ELTYPE;
    static constexpr auto el_o  = ELORDER;
};

template <typename T>
void initializeElement(T)
{
    auto op = [](std::pair<ElementTypes, types::el_o_t> in)
    {
        return in == std::make_pair(T::el_t, T::el_o);
    };

    if (std::none_of(ElementInitializerMaster::initialized_elements.cbegin(),
                     ElementInitializerMaster::initialized_elements.cend(), op))
    {
        // Add to list of initialized elements
        ElementInitializerMaster::
        initialized_elements.emplace_back(T::el_t, T::el_o);

        // Register element vector with appropriate factory
        util::Factory< std::pair<ElementTypes, types::el_o_t>, ElementVectorBase >::
        registerCreator < ElementVector<T::el_t, T::el_o> >
        (std::make_pair(T::el_t, T::el_o));

        util::Factory< std::pair<ElementTypes, types::el_o_t>, ElementBase >::
        registerCreator < Element<T::el_t, T::el_o> >(std::make_pair(T::el_t, T::el_o));

        // initialize element of same type, order 1
        initializeElement(ElementInitializer<T::el_t, 1> {});
    }
}
}           // namespace mesh
}           // namespace lstr

#endif      // end include guard
