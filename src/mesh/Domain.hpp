// Domain handling

#ifndef L3STER_INCGUARD_MESH_DOMAIN_HPP
#define L3STER_INCGUARD_MESH_DOMAIN_HPP

#include "mesh/ElementVector.hpp"
#include "utility/Factory.hpp"

#include <map>
#include <memory>
#include <utility>
#include <stdexcept>

namespace lstr
{
namespace mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                                      DOMAIN CLASS                                        //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
The domain class stores 1+ element vectors
*/
class Domain
{
public:
    using map_t = std::map< std::pair<ElementTypes, types::el_o_t>, std::unique_ptr<ElementVectorBase> >;

    types::d_id_t       getId() const               { return id; }
    void                setId(types::d_id_t _id)    { id = _id; }

    template <ElementTypes ELTYPE, types::el_o_t ELORDER>
    void pushBack(const Element<ELTYPE, ELORDER>&);

    template <ElementTypes ELTYPE, types::el_o_t ELORDER, typename ... Types>
    void emplaceBack(Types&& ... Args);

private:
    types::d_id_t       id          = 0;
    map_t               element_vectors;
};

// Proxy for puch_back of the underlying vector
template <ElementTypes ELTYPE, types::el_o_t ELORDER>
void Domain::pushBack(const Element<ELTYPE, ELORDER>& element)
{
    auto pos_iter = element_vectors.find(std::make_pair(ELTYPE, ELORDER));

    // If vector of elements of given type does not exist, create it
    if (pos_iter == element_vectors.end())
    {
        auto insert_result = element_vectors.insert(std::make_pair(std::make_pair(ELTYPE, ELORDER),
                             std::make_unique< ElementVector<ELTYPE, ELORDER> >()));

        if (!insert_result.second)
        {
            throw (std::runtime_error("Element insertion failed\n"));
        }

        pos_iter = insert_result.first;
    }

    // Push element back to appropriate vector
    static_cast< ElementVector<ELTYPE, ELORDER>* >(pos_iter->second.get())->getRef().push_back(element);
}

// Proxy for emplace_back of the underlying vector
template <ElementTypes ELTYPE, types::el_o_t ELORDER, typename ... Types>
void Domain::emplaceBack(Types&& ... Args)
{
    auto pos_iter = element_vectors.find(std::make_pair(ELTYPE, ELORDER));

    // If vector of elements of given type does not exist, create it
    if (pos_iter == element_vectors.end())
    {
        auto insert_result = element_vectors.insert(std::make_pair(std::make_pair(ELTYPE, ELORDER),
                             std::make_unique< ElementVector<ELTYPE, ELORDER> >()));

        if (!insert_result.second)
        {
            throw (std::runtime_error("Element insertion failed\n"));
        }

        pos_iter = insert_result.first;
    }

    // Push element back to appropriate vector
    static_cast< ElementVector<ELTYPE, ELORDER>* >(pos_iter->second.get())->getRef().emplace_back(std::forward<Types>(Args) ...);
}
}
}

#endif      // end include guard
