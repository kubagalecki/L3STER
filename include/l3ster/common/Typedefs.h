#ifndef L3STER_COMMON_TYPEDEFS_H
#define L3STER_COMMON_TYPEDEFS_H

#include <cstdint>
#include <limits>

namespace lstr
{
using std::ptrdiff_t;
using std::size_t;

using dim_t        = std::uint_fast8_t; // spatial dimension type
using n_id_t       = std::uint64_t;     // node ID type
using n_loc_id_t   = std::uint32_t;     // local node ID type
using el_id_t      = std::uint64_t;     // element id type
using el_loc_id_t  = std::uint32_t;     // element id type
using el_o_t       = std::uint8_t;      // element order type
using el_side_t    = std::uint8_t;      // index of element side (points/edges/faces) type
using el_locind_t  = std::uint16_t;     // local element index: nodes/element must not overflow this
using d_id_t       = std::uint16_t;     // domain id type
using q_o_t        = std::uint8_t;      // quadrature order type
using q_l_t        = std::uint64_t;     // quadrature length type
using val_t        = double;            // floating point value type
using global_dof_t = long long int;     // global DOF type
using local_dof_t  = int;               // local DOF type

inline constexpr auto         invalid_domain_id  = std::numeric_limits< d_id_t >::max();
inline constexpr auto         invalid_dim        = std::numeric_limits< dim_t >::max();
inline constexpr auto         invalid_node       = std::numeric_limits< n_id_t >::max();
inline constexpr global_dof_t invalid_global_dof = -1;
inline constexpr local_dof_t  invalid_local_dof  = -1;
} // namespace lstr
#endif // L3STER_COMMON_TYPEDEFS_H
