#ifndef L3STER_TESTS_COMMON_HPP
#define L3STER_TESTS_COMMON_HPP

#include <iostream>
#include <source_location>
#include <sstream>

inline void logErrorAndTerminate(std::string_view err_msg, std::source_location sl = std::source_location::current())
{
    std::stringstream err_msg_str;
    err_msg_str << sl.file_name() << '(' << sl.line() << ", " << sl.column() << ')' << " in function "
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
#endif // L3STER_TESTS_COMMON_HPP
