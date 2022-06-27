#ifndef L3STER_ASSEMBLY_BOUNDARYSYSTEMASSEMBLY_HPP
#define L3STER_ASSEMBLY_BOUNDARYSYSTEMASSEMBLY_HPP

#include "l3ster/assembly/AssembleLocalSystem.hpp"

namespace lstr
{
template < typename Kernel, ElementTypes ET, el_o_t EO, q_l_t QL, int n_fields >
auto assembleBoundarySystem(Kernel&&                                                                       kernel,
                            const Element< ET, EO >&                                                       element,
                            const Eigen::Matrix< val_t, Element< ET, EO >::n_nodes, n_fields >&            node_vals,
                            const ReferenceBasisAtQuadrature< ET, EO, QL, Element< ET, EO >::native_dim >& basis_at_q,
                            val_t                                                                          time)
    requires detail::Kernel_c< Kernel, ET, EO, n_fields >
{}
} // namespace lstr
#endif // L3STER_ASSEMBLY_BOUNDARYSYSTEMASSEMBLY_HPP
