CMAKE_MINIMUM_REQUIRED(VERSION 3.15)

PROJECT(L3STER_tests)

set(CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_BUILD_TYPE RELEASE)

if (UNIX)
    add_compile_options(-Wall -Wextra -Wpedantic -Wshadow)
elseif(MSVC)
    add_compile_options(/Wall /experimental:newLambdaProcessor)
endif()

# SRC
include_directories(${CMAKE_SOURCE_DIR}/../include/)

# Catch2
include_directories(${CMAKE_SOURCE_DIR}/../Catch2/single_include/)

add_executable(test
        ${CMAKE_SOURCE_DIR}/tests_main.cpp
	    ${CMAKE_SOURCE_DIR}/quadrature_test.cpp
	)

