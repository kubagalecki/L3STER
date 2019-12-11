// Class for forcing the compilation of specific quadratures

#ifndef L3STER_INCGUARD_QUADRATURE_QUADRATUREINITIALIZER_HPP
#define L3STER_INCGUARD_QUADRATURE_QUADRATUREINITIALIZER_HPP

#include "mesh/ElementTypes.h"
#include "typedefs/Types.h"
#include "quadrature/Quadrature.hpp"
#include "quadrature/QuadratureGenerator.hpp"

namespace lstr
{
namespace quad
{
//////////////////////////////////////////////////////////////////////////////////////////////
//                              QUADRATURE INITIALIZER CLASS                                //
//////////////////////////////////////////////////////////////////////////////////////////////
/*
Helper class for initializing specific quadratures
*/
template <QuadratureTypes QTYPE, types::q_o_t QORDER>
class QuadratureInitializer
{
public:
    // Aliases
    using q_pair_t = std::pair<quad::QuadratureTypes, types::q_o_t>;

private:
    static std::map< q_pair_t, std::unique_ptr<quad::QuadratureBase> > quadratures;
};
}           // namespace quad
}           // namespace lstr

#endif      // end include guard
