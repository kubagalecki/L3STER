#ifndef L3STER_SIMSTRUCTURE_MESHDEF_HPP
#define L3STER_SIMSTRUCTURE_MESHDEF_HPP

#include "defs/Typedefs.h"

#include <string_view>

namespace lstr::def
{
enum struct MeshFormat
{
    Gmsh
};

struct Mesh
{
    constexpr Mesh(std::string_view name_, el_o_t order_ = 2u, MeshFormat format_ = MeshFormat::Gmsh)
        : name{name_}, format{format_}, order{order_}
    {}

    std::string_view name;
    MeshFormat       format;
    el_o_t           order;
};
} // namespace lstr::def
#endif // L3STER_SIMSTRUCTURE_MESHDEF_HPP
