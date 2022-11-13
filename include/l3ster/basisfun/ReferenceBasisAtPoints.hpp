#ifndef L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
#define L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP

#include "l3ster/mesh/ElementTraits.hpp"
#include "l3ster/util/IncludeEigen.hpp"

namespace lstr
{
template < ElementTypes ET, el_o_t EO, size_t n_points >
struct ReferenceBasisAtPoints
{
private:
    using basis_at_qp_t = EigenRowMajorMatrix< val_t, n_points, Element< ET, EO >::n_nodes >;
    using basis_ders_t  = std::array< basis_at_qp_t, Element< ET, EO >::native_dim >;

public:
    basis_at_qp_t values;
    basis_ders_t  derivatives;
};
} // namespace lstr
#endif // L3STER_BASISFUN_REFERENCEBASISATPOINTS_HPP
