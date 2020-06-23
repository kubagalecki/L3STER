#include "mesh/Mesh.hpp"
#include "mesh/readMesh.hpp"
#include "quadrature/Quadrature.hpp"

#include <iostream>

using namespace lstr;

int main()
{
    try
    {
        mesh::Mesh m = mesh::readMesh("/home/jgalecki/Documents/Sandbox/gmsh-sample-msh/test4.msh", mesh::gmsh_tag);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    catch (...)
    {
        std::cerr << "Unknown exception was thrown\n";
    }
    quad::Quadrature< 1, 1 > q;
}
