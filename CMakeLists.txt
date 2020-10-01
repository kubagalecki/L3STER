cmake_minimum_required(VERSION 3.13)

project(L3STER LANGUAGES CXX VERSION 0.1.0)

###  Find dependencies  ###
# Find Eigen
find_package (Eigen3 3.3 REQUIRED NO_MODULE)

# Detect if tests are to be built, if yes then try to find Catch2
if(L3STER_ENABLE_TESTS)
    message(STATUS "Tests are enabled. Finding Catch2...")

    # Try to find if installed
    message(STATUS "Looking for Catch2 using find_package...")
    find_package(Catch2)

    if(TARGET Catch2::Catch2)
        message(STATUS "Looking for Catch2 using find_package... FOUND")
    else()
        message(STATUS "Looking for Catch2 using find_package... NOT FOUND")

        message(STATUS "Looking for Catch2 in the source directory...")
        if((EXISTS ${CMAKE_CURRENT_SOURCE_DIR}/Catch2) AND (IS_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/Catch2))
            message(STATUS "Looking for Catch2 in the source directory... FOUND")
            add_subdirectory(Catch2)
        else()
            message(STATUS "Looking for Catch2 in the source directory... NOT FOUND")
            message(FATAL_ERROR "Could not find Catch2")
        endif()
    endif()
endif()

message(STATUS "Creating L3STER interface target...")
add_library(L3STER INTERFACE)
target_include_directories(L3STER INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/include/)
file(GLOB_RECURSE L3STER_SOURCES
    LIST_DIRECTORIES false
    RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
    CONFIGURE_DEPENDS
    include/*
)
target_sources(L3STER INTERFACE ${L3STER_SOURCES})
set_target_properties(L3STER PROPERTIES
    INTERFACE_COMPILE_FEATURES cxx_std_20
)
target_link_libraries(L3STER INTERFACE
    Eigen3::Eigen
)
message(STATUS "Creating L3STER interface target... DONE")

if(L3STER_ENABLE_TESTS)
    message(STATUS "Creating L3STER test target...")
    add_executable(L3STER_tests
        tests/TestsMain.cpp
        tests/QuadratureTests.cpp
	    tests/MeshTests.cpp
    )
    set(L3STER_TESTS_DATA_PATH ${CMAKE_CURRENT_SOURCE_DIR}/tests/data CACHE STRING "Path to test data directory")
    configure_file(tests/TestDataPath.h.in TestDataPath.h)
    target_include_directories(L3STER_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    target_compile_options(L3STER_tests PUBLIC
        -Wall
        -Wextra
        -Wpedantic
        -Wshadow
        -fprofile-arcs
        -ftest-coverage
        -fPIC
    )
    target_link_options(L3STER_tests PUBLIC
        -lgcov
        --coverage
    )
    target_link_libraries(L3STER_tests
        L3STER
        Catch2::Catch2
    )
    message(STATUS "Creating L3STER test target... DONE")

    message(STATUS "Registering tests with CTest...")
    include(CTest)
    include(Catch)
    catch_discover_tests(L3STER_tests)
    message(STATUS "Registering tests with CTest... DONE")
endif()
