#include "mesh/MeshMaster.hpp"
#include "quadrature/QuadratureGenerator.hpp"
#include "utility/Polynomial.hpp"

# include "utility/Factory.hpp"

#include <iostream>
#include <exception>

int main()
{
    std::vector<int> v = {1, 2, 3};
    for (int i = 0; const auto& e : v)
        std::cout << i++ << ' ' << e << '\n';
    
    try
    {
        auto&& q = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1> { {1, 2, 8, 6} };
        auto&& d = lstr::mesh::Domain{};
        auto&& NA = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1>::node_array_t{1, 2, 8, 6};
        d.emplaceBack<lstr::mesh::ElementTypes::Quad, 1>(std::move(NA));

        auto&& Q = lstr::quad::QuadratureGenerator<lstr::mesh::ElementTypes::Quad>::
        getQuadrature<lstr::quad::QuadratureTypes::GLeg, 2>();
        
        using a_t = std::array<lstr::types::val_t, 3>;
        const auto x = a_t{0., 1., -1.};
        const auto y = a_t{0., 1., 1.};
        auto p2 = lstr::util::lagrangeFit(x, y);

        auto ev =
        lstr::util::Factory< std::pair<lstr::mesh::ElementTypes, lstr::types::el_o_t>, lstr::mesh::ElementVectorBase >
        ::create(std::make_pair(lstr::mesh::ElementTypes::Quad, 1));
        
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

