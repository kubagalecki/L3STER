#include "l3ster.hpp"

#include <iomanip>
#include <iostream>

using namespace lstr;

int main()
{
    const auto mesh = mesh::readMesh("../tests/data/mesh_ascii4.msh", mesh::gmsh_tag);

    std::vector< mesh::BoundaryView > boundaries;
    boundaries.reserve(4);
    for (int i = 2; i <= 5; ++i)
        boundaries.emplace_back(mesh.getPartitions()[0], i);

    const auto boundary_visitor = [&nodes = mesh.getNodes()](const auto& bev) {
        std::cout << static_cast< int >(bev.element_side) << ", \t";
        for (const auto& node : bev.element.get().getNodes())
            std::cout << node << '\t';
        std::cout << '\n';
    };

    for (const auto& boundary_view : boundaries)
        boundary_view.visit(boundary_visitor);

    return EXIT_SUCCESS;
}
