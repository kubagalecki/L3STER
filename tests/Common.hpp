#ifndef L3STER_TESTS_COMMON_HPP
#define L3STER_TESTS_COMMON_HPP

#define CHECK_THROWS(EXPR)                                                                                             \
    try                                                                                                                \
    {                                                                                                                  \
        EXPR;                                                                                                          \
        return EXIT_FAILURE;                                                                                           \
    }                                                                                                                  \
    catch (...)                                                                                                        \
    {}

#endif // L3STER_TESTS_COMMON_HPP
