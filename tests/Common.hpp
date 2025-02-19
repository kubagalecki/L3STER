#ifndef L3STER_TESTS_COMMON_HPP
#define L3STER_TESTS_COMMON_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Assertion.hpp"

#include <iostream>
#include <print>

inline void logErrorAndTerminate(std::string_view     err_msg,
                                 std::source_location src_loc = std::source_location::current())
{
    std::cerr << lstr::util::detail::makeErrMsg(err_msg, src_loc) << '\n';
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
void summarizeMesh(const lstr::MpiComm& comm, const lstr::mesh::MeshPartition< orders... >& mesh)
{
    for (int rank = 0; rank < comm.getSize(); ++rank)
    {
        if (comm.getRank() == rank)
        {
            std::println("*** Rank {} ***\nOwned nodes: {}, ghost nodes: {}",
                         comm.getRank(),
                         mesh.getNodeOwnership().owned().size(),
                         mesh.getNodeOwnership().shared().size(),
                         mesh.getNDomains());
            constexpr auto fmt_str = "{:^12}|{:^12}|{:^12}";
            std::println(fmt_str, "Domain ID", "Dimension", "#Elements");
            for (auto dom : mesh.getDomainIds())
                std::println(fmt_str, dom, mesh.getDomain(dom).dim, mesh.getDomain(dom).numElements());
            std::println();
            if (comm.getRank() == comm.getSize() - 1)
                std::println("---\n");
        }
        comm.barrier();
    }
}

template < lstr::el_o_t... orders >
void printMesh(const lstr::MpiComm& comm, const lstr::mesh::MeshPartition< orders... >& mesh)
{
    for (int rank = 0; rank < comm.getSize(); ++rank)
    {
        if (comm.getRank() == rank)
        {
            std::print("Rank: {}\nOwned nodes: ", comm.getRank());
            for (auto n : mesh.getNodeOwnership().owned())
                std::print("{} ", n);
            std::print("\nGhost nodes: ");
            for (auto n : mesh.getNodeOwnership().shared())
                std::print("{} ", n);
            std::println();
            for (auto dom : mesh.getDomainIds())
            {
                std::println("Domain: {}", dom);
                mesh.visit(
                    []< lstr::mesh::ElementType ET, lstr::el_o_t EO >(const lstr::mesh::Element< ET, EO >& element) {
                        std::print("Element ID: {}, type: {}, order: {:d}, nodes: ",
                                   element.getId(),
                                   std::to_underlying(ET),
                                   EO);
                        for (auto n : element.getNodes())
                            std::print("{} ", n);
                        std::println();
                    },
                    dom);
            }
            std::println();
            if (comm.getRank() == comm.getSize() - 1)
                std::println("---\n");
        }
        comm.barrier();
    }
}
#endif // L3STER_TESTS_COMMON_HPP
