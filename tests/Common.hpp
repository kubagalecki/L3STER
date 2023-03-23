#ifndef L3STER_TESTS_COMMON_HPP
#define L3STER_TESTS_COMMON_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/MeshPartition.hpp"

#include <iostream>
#include <source_location>
#include <sstream>

inline void logErrorAndTerminate(std::string_view err_msg, std::source_location sl = std::source_location::current())
{
    std::stringstream err_msg_str;
    err_msg_str << sl.file_name() << ':' << sl.line() << ':' << sl.column() << ':' << "\nIn function "
                << sl.function_name() << ": " << err_msg << '\n';
    std::cerr << err_msg_str.view();
    std::terminate();
}

#define REQUIRE(EXPR)                                                                                                  \
    if (not(EXPR))                                                                                                     \
    logErrorAndTerminate("The following expression evaluated to false:\n" #EXPR)

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

inline void describeMesh(const lstr::MpiComm& comm, const lstr::MeshPartition& mesh)
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
                    []< lstr::ElementTypes ET, lstr::el_o_t EO >(const lstr::Element< ET, EO >& element) {
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
