// ReferenceElement class implementation

#ifndef L3STER_INCGUARD_MESH_REFERENCEELEMENT_HPP
#define L3STER_INCGUARD_MESH_REFERENCEELEMENT_HPP

#include "mesh/ElementTypes.hpp"
#include "quadrature/Quadrature.hpp"
#include "typedefs/Types.h"

#include <map>
#include <memory>
#include <utility>

namespace lstr
{
namespace mesh
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                             REFERENCE ELEMENT TRAITS CLASS                               //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
This class is to be specialized for each element type. It must contain the following member
functions:

static constexpr size_t            getNumberOfNodes(types::el_o_t)

static constexpr types::el_dim_t   getDim()

template <quad::QuadratureTypes QTYPE>
static constexpr size_t             getQuadratureSize(types::q_o_t);
*/
template < ElementTypes ELTYPE >
struct ReferenceElementTraits;

//////////////////////////////////////////////////////////////////////////////////////////////
//                              REFERENCE BASE ELEMENT CLASS                                //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Static base class for ReferenceElement, contains all element type-specific data, makes use of
ReferenceElementTraits
*/
template < ElementTypes ELTYPE >
class ReferenceElementBase
{
public:
    // Aliases
    using q_pair_t = std::pair< quad::QuadratureTypes, types::q_o_t >;

    // Info access
    static constexpr size_t getNumberOfNodes(types::el_o_t);

    static constexpr types::el_dim_t getDim();

    static constexpr size_t getQuadratureSize(quad::QuadratureTypes, types::q_o_t);

    // ReferenceElementBase is a static class
    ReferenceElementBase() = delete;

    ReferenceElementBase(const ReferenceElementBase&) = delete;

    ReferenceElementBase& operator=(const ReferenceElementBase&) = delete;

    ~ReferenceElementBase() = delete;

    ReferenceElementBase(const ReferenceElementBase&&) = delete;

    ReferenceElementBase& operator=(const ReferenceElementBase&&) = delete;

private:
    static std::map< q_pair_t, std::unique_ptr< quad::QuadratureBase > > quadratures;
};

template < ElementTypes ELTYPE >
constexpr size_t ReferenceElementBase< ELTYPE >::getNumberOfNodes(types::el_o_t ELORDER)
{
    return ReferenceElementTraits< ELTYPE >::getNumberOfNodes(ELORDER);
}

template < ElementTypes ELTYPE >
constexpr types::el_dim_t ReferenceElementBase< ELTYPE >::getDim()
{
    return ReferenceElementTraits< ELTYPE >::getDim();
}

template < ElementTypes ELTYPE >
constexpr size_t ReferenceElementBase< ELTYPE >::getQuadratureSize(quad::QuadratureTypes qtype,
                                                                   types::q_o_t          qorder)
{
    return ReferenceElementTraits< ELTYPE >::getQuadratureSize(qtype, qorder);
}

//////////////////////////////////////////////////////////////////////////////////////////////
//                                REFERENCE ELEMENT CLASS                                   //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
This static class constitutes the interface through which general information about elements of
a given type and order is available. The interface is available within the constexpr context.
*/
template < ElementTypes ELTYPE, types::el_o_t ELORDER >
class ReferenceElement final : ReferenceElementBase< ELTYPE >
{
public:
    // ReferenceElement is a static class
    ReferenceElement() = delete;

    ReferenceElement(const ReferenceElement&) = delete;

    ReferenceElement& operator=(const ReferenceElement&) = delete;

    ~ReferenceElement() = delete;

    ReferenceElement(const ReferenceElement&&) = delete;

    ReferenceElement& operator=(const ReferenceElement&&) = delete;

    // METHODS
    static constexpr size_t getNumberOfNodes() { return parent_t::getNumberOfNodes(ELORDER); }

    static constexpr types::el_dim_t getDim() { return parent_t::getDim(); }

private:
    using parent_t = ReferenceElementBase< ELTYPE >;
};

//////////////////////////////////////////////////////////////////////////////////////////////
//                         REFERENCE ELEMENT TRAITS SPECIALIZATIONS                         //
//////////////////////////////////////////////////////////////////////////////////////////////
// When adding new element types, modify this file only below this point

// QUAD
template <>
struct ReferenceElementTraits< ElementTypes::Quad > final
{
    static constexpr size_t getNumberOfNodes(types::el_o_t elorder)
    {
        return (elorder + 1) * (elorder + 1);
    }

    static constexpr types::el_dim_t getDim() { return 2; }
};
//////////////////////////////////////////////////////////////////////////////////////////////
} // namespace mesh
} // namespace lstr

#endif // end include guard
