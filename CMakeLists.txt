cmake_minimum_required(VERSION 3.15)

project(L3STER LANGUAGES CXX VERSION 0.1.0)

set(L3STER_DIR ${CMAKE_CURRENT_SOURCE_DIR})

###  Find dependencies  ###
set(L3STER_Verbosity ON) # Report dependency detection progress
include("${L3STER_DIR}/scripts/cmake/DetectDepends.cmake")

include(scripts/cmake/MakeL3STERTarget.cmake)

if (L3STER_ENABLE_TESTS)
    include(scripts/cmake/MakeL3STERTestTarget.cmake)
endif ()
