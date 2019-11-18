#include "MeshMaster.hpp"
#include "Polynomial.hpp"
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
        
        using a_t = std::array<lstr::types::val_t, 2>;
        const auto x = std::array<lstr::types::val_t, 2>{0., 1.};
        const auto y = std::array<lstr::types::val_t, 2>{0., 1.};
        auto p2 = lstr::util::lagrangeFit(x, y);
        
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
