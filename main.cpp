#include "l3ster.hpp"

#include <iomanip>
#include <iostream>

using namespace lstr;

struct StatefulVerboseElementPrinter
{
    StatefulVerboseElementPrinter() { std::cerr << "default ctor\n"; }
    StatefulVerboseElementPrinter(const StatefulVerboseElementPrinter& r) = delete;
    //{
    //    counter = r.counter;
    //    std::cerr << "copy\n";
    //}
    StatefulVerboseElementPrinter(StatefulVerboseElementPrinter&& r) noexcept
    {
        counter = r.counter;
        std::cerr << "move\n";
    }
    StatefulVerboseElementPrinter& operator=(const StatefulVerboseElementPrinter& r) = delete;
    //{
    //    counter = r.counter;
    //    std::cerr << "copy=\n";
    //    return *this;
    //}
    StatefulVerboseElementPrinter& operator=(StatefulVerboseElementPrinter&& r) noexcept
    {
        counter = r.counter;
        std::cerr << "move=\n";
        return *this;
    }
    ~StatefulVerboseElementPrinter() { std::cerr << "dtor\n"; }

    template < mesh::ElementTypes ELTYPE, types::el_o_t ELORDER >
    void operator()(const mesh::Element< ELTYPE, ELORDER >&)
    {
        ++counter;
    }

    size_t counter = 0;
};

int main()
{
    mesh::Mesh m;
    try
    {
        m = mesh::readMesh("../../../tests/data/mesh_ascii4.msh", mesh::gmsh_tag);
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
        return EXIT_FAILURE;
    }
    catch (...)
    {
        std::cerr << "Unknown exception was thrown\n";
        return EXIT_FAILURE;
    }

    std::cout << std::fixed << std::setprecision(4);

    // m.getPartitions()[0].cvisitSpecifiedDomains(
    //    [&m, counter = 0u](const auto& element) mutable {
    //        std::for_each(
    //            element.getNodes().cbegin(), element.getNodes().cend(), [&m](const auto& n) {
    //                std::for_each(m.getNodes()[n].coords.cbegin(),
    //                              m.getNodes()[n].coords.cend(),
    //                              [](const auto& c) { std::cout << c << '\t'; });
    //                std::cout << '\n';
    //            });
    //        std::cout << '\n';
    //        ++counter;
    //    },
    //    {1});

    //auto v = StatefulVerboseElementPrinter{};

    //v = m.getPartitions()[0].cvisitAllElements(std::move(v));
    //v = m.getPartitions()[0].visitAllElements(std::move(v));

    //std::cout << v.counter << '\n';

    const auto element_predicate = [](const auto& element) {
        if constexpr (mesh::ElementTraits<std::decay_t < decltype(element) >>::element_type == mesh::ElementTypes::Quad)
        {
            return std::find(element.getNodes().cbegin(), element.getNodes().cend(), 0) !=
                   element.getNodes().cend();
        }
        else
        {
            return false;
        }
    };

    const auto el = m.getPartitions()[0].findElement(element_predicate);

    if (!el)
        return EXIT_FAILURE;

    std::visit(
        [](const auto& element) {
            std::for_each(element.get().getNodes().cbegin(),
                          element.get().getNodes().cend(),
                          [](const auto n) {
                std::cout << n << '\t';
            });
            std::cout << '\n';
        },
        *el);

    return EXIT_SUCCESS;
}
