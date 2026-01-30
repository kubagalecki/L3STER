#ifndef L3STER_MAPPING_REFERENCEBOUNDARYTOSIDEMAPPING_HPP
#define L3STER_MAPPING_REFERENCEBOUNDARYTOSIDEMAPPING_HPP

#include "l3ster/math/RotationMatrix.hpp"
#include "l3ster/mesh/ElementTraits.hpp"

#include <numbers>
#include <utility>

namespace lstr::map
{
// Get the mapping from the reference boundary element to the reference element side. E.g. map the 2D reference quad
// onto the side of the 3D reference hex. This mapping is linear and can be broken down into a rotation and translation.
template < mesh::ElementType ET >
const auto& getReferenceBoundaryToSideMapping(el_side_t side)
    requires(mesh::isGeomType(ET))
{
    static const auto mapping_lookup_table = std::invoke([] {
        using traits              = mesh::ElementTraits< mesh::Element< ET, 1 > >;
        constexpr auto native_dim = traits::native_dim;
        constexpr auto n_sides    = traits::n_sides;
        using rot_mat_t           = Eigen::Matrix< val_t, native_dim, native_dim >;
        using trans_vec_t         = Eigen::Vector< val_t, native_dim >;
        using std::numbers::pi;
        using enum mesh::ElementType;
        std::array< std::pair< rot_mat_t, trans_vec_t >, n_sides > table;
        if constexpr (ET == Hex)
        {
            table[0] = std::make_pair(math::makeRotationMatrix3DX(pi), trans_vec_t(0., 0., -1.));
            table[1] = std::make_pair(rot_mat_t::Identity(), trans_vec_t(0., 0., 1.));
            table[2] = std::make_pair(math::makeRotationMatrix3DX(-pi / 2.), trans_vec_t(0., -1., 0.));
            table[3] = std::make_pair(math::makeRotationMatrix3DX(pi / 2.), trans_vec_t(0., 1., 0.));
            table[4] = std::make_pair(math::makeRotationMatrix3DY(pi / 2.), trans_vec_t(-1., 0., 0.));
            table[5] = std::make_pair(math::makeRotationMatrix3DY(-pi / 2.), trans_vec_t(1., 0., 0.));
        }
        else if constexpr (ET == Quad)
        {
            table[0] = std::make_pair(math::makeRotationMatrix2D(pi), trans_vec_t(0., -1.));
            table[1] = std::make_pair(rot_mat_t::Identity(), trans_vec_t(0., 1.));
            table[2] = std::make_pair(math::makeRotationMatrix2D(pi / 2), trans_vec_t(-1., 0.));
            table[3] = std::make_pair(math::makeRotationMatrix2D(-pi / 2), trans_vec_t(1., 0.));
        }
        else if constexpr (ET == Line)
        {
            table[0] = std::make_pair(-rot_mat_t::Ones(), trans_vec_t(-1.));
            table[1] = std::make_pair(rot_mat_t::Ones(), trans_vec_t(1.));
        }
        else
            static_assert(util::always_false< ET >);
        return table;
    });
    return mapping_lookup_table[side];
}
} // namespace lstr::map
#endif // L3STER_MAPPING_REFERENCEBOUNDARYTOSIDEMAPPING_HPP
