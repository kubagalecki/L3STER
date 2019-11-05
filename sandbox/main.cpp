#include "Domain.hpp"
#include <iostream>

int main()
{
    try
    {
        //auto&& q = lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1> { {1, 2, 8, 6} };
        auto&& d = lstr::mesh::Domain{};
        auto&& NA = typename lstr::mesh::Element<lstr::mesh::ElementTypes::Quad, 1>::node_array_t{1, 2, 8, 6};
        d.emplaceBack<lstr::mesh::ElementTypes::Quad, 1>(NA);
        std::cout << "all done!\n";
        return 0;
    }
    catch (const std::exception& e)
    {
        std::cout << e.what() << std::endl;
        return 1;
    }
    catch (...)
    {
        std::cout << "Unknown exception was thrown" << std::endl;
        return 2;
    }
}
