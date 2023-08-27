#ifndef L3STER_TESTS_COMMON_HPP
#define L3STER_TESTS_COMMON_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/BoundaryView.hpp"
#include "l3ster/util/Assertion.hpp"

#include <iostream>

inline void logErrorAndTerminate(std::string_view     err_msg,
                                 std::source_location src_loc = std::source_location::current())
{
    auto printed_error = lstr::util::detail::parseSourceLocation(src_loc);
    std::ranges::copy(err_msg, std::back_inserter(printed_error));
    std::cerr << printed_error << '\n';
    std::terminate();
}

#define REQUIRE(EXPR)                                                                                                  \
    if (not(EXPR))                                                                                                     \
    logErrorAndTerminate("The following expression evaluated to false:\n\n  " #EXPR "\n\n")

#define CHECK_THROWS(EXPR)                                                                                             \
    std::invoke([&] {                                                                                                  \
        struct Except                                                                                                  \
        {};                                                                                                            \
        try                                                                                                            \
        {                                                                                                              \
            EXPR;                                                                                                      \
            throw Except{};                                                                                            \
        }                                                                                                              \
        catch (const Except&)                                                                                          \
        {                                                                                                              \
            throw std::runtime_error{"Expression which was expected to throw failed to do so"};                        \
        }                                                                                                              \
        catch (...)                                                                                                    \
        {}                                                                                                             \
    })

template < lstr::el_o_t... orders >
void describeMesh(const lstr::MpiComm& comm, const lstr::mesh::MeshPartition< orders... >& mesh)
{
    for (int rank = 0; rank < comm.getSize(); ++rank)
    {
        if (comm.getRank() == rank)
        {
            std::cout << "Rank: " << comm.getRank() << "\nOwned nodes: ";
            for (auto n : mesh.getOwnedNodes())
                std::cout << n << ' ';
            std::cout << "\nGhost nodes: ";
            for (auto n : mesh.getGhostNodes())
                std::cout << n << ' ';
            std::cout << '\n';
            for (auto dom : mesh.getDomainIds())
            {
                std::cout << "Domain: " << dom << '\n';
                mesh.visit(
                    []< lstr::mesh::ElementType ET, lstr::el_o_t EO >(const lstr::mesh::Element< ET, EO >& element) {
                        std::cout << "Element ID: " << element.getId() << ", type: " << static_cast< int >(ET)
                                  << ", order: " << static_cast< int >(EO) << ", nodes: ";
                        for (auto n : element.getNodes())
                            std::cout << n << ' ';
                        std::cout << '\n';
                    },
                    dom);
            }
            std::cout << std::endl;
        }
        comm.barrier();
    }
}
#endif // L3STER_TESTS_COMMON_HPP
