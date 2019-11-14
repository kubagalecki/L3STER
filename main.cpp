#include "Domain.hpp"
#include <iostream>
#include <exception>

int main()
{
    try
    {
        auto&& q = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1> { {1, 2, 8, 6} };
        auto&& d = lstr::mesh::Domain{};
        auto&& NA = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1>::node_array_t{1, 2, 8, 6};
        d.emplaceBack<lstr::mesh::ElementTypes::Quad, 1>(NA);

        auto&& Q = lstr::quad::Quadrature<lstr::mesh::ElementTypes::Quad, lstr::quad::QuadratureTypes::GLeg, 2>{};
        
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cerr << "Unknown exception was thrown" << std::endl;
        return 2;
    }
}
