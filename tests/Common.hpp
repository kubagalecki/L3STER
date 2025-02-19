#ifndef L3STER_TESTS_COMMON_HPP
#define L3STER_TESTS_COMMON_HPP

#include "l3ster/comm/MpiComm.hpp"
#include "l3ster/mesh/MeshPartition.hpp"
#include "l3ster/util/Assertion.hpp"

#include <format>
#include <iostream>

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
            std::cout << std::format("*** Rank {} ***\nOwned nodes: {}, ghost nodes: {}\n",
                                     comm.getRank(),
                                     mesh.getNodeOwnership().owned().size(),
                                     mesh.getNodeOwnership().shared().size(),
                                     mesh.getNDomains());
            constexpr auto fmt_str = "{:^12}|{:^12}|{:^12}\n";
            std::cout << std::format(fmt_str, "Domain ID", "Dimension", "#Elements");
            for (auto dom : mesh.getDomainIds())
                std::cout << std::format(fmt_str, dom, mesh.getDomain(dom).dim, mesh.getDomain(dom).numElements());
            std::cout << std::endl;
            if (comm.getRank() == comm.getSize() - 1)
                std::cout << "---\n" << std::endl;
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
            std::cout << std::format("Rank: {}\nOwned nodes: ", comm.getRank());
            for (auto n : mesh.getNodeOwnership().owned())
                std::cout << std::format("{} ", n);
            std::cout << "\nGhost nodes: ";
            for (auto n : mesh.getNodeOwnership().shared())
                std::cout << std::format("{} ", n);
            std::cout << '\n';
            for (auto dom : mesh.getDomainIds())
            {
                std::cout << std::format("Domain: {}\n", dom);
                mesh.visit(
                    []< lstr::mesh::ElementType ET, lstr::el_o_t EO >(const lstr::mesh::Element< ET, EO >& element) {
                        std::cout << std::format("Element ID: {}, type: {}, order: {:d}, nodes: ",
                                                 element.getId(),
                                                 std::to_underlying(ET),
                                                 EO);
                        for (auto n : element.getNodes())
                            std::cout << std::format("{} ", n);
                        std::cout << '\n';
                    },
                    dom);
            }
            std::cout << std::endl;
            if (comm.getRank() == comm.getSize() - 1)
                std::cout << "---\n" << std::endl;
        }
        comm.barrier();
    }
}
#endif // L3STER_TESTS_COMMON_HPP
