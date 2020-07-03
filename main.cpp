#include "l3ster.h"

#include <iostream>

using namespace lstr;

int main()
{
    std::shared_ptr< mesh::Mesh > m;
    try
    {
        m = mesh::readMesh("/home/jgalecki/Documents/Sandbox/gmsh-sample-msh/test4.msh",
                           mesh::gmsh_tag);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    catch (...)
    {
        std::cerr << "Unknown exception was thrown\n";
    }

    m->getPartitionsRef()[0].visitAllElements([&m](const auto& element) {
        std::for_each(element.getNodes().cbegin(), element.getNodes().cend(), [&m](const auto& n) {
            std::for_each(m->getNodes()[n].coords.cbegin(),
                          m->getNodes()[n].coords.cend(),
                          [](const auto& c) { std::cout << c << '\t'; });
            std::cout << '\n';
        });
        std::cout << '\n';
    });

    return EXIT_SUCCESS;
}
