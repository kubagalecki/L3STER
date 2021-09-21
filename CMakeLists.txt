cmake_minimum_required( VERSION 3.13 )
project( L3STER LANGUAGES CXX VERSION 0.1.0 )
#######################################################################################################################

# Require Linux

if ( NOT ${CMAKE_SYSTEM_NAME} STREQUAL Linux )
    message( FATAL_ERROR "Currently, L3STER only supports Linux" )
endif ()

#######################################################################################################################

# L3STER options

# Verbosity (mainly relating to finding dependencies)
option( L3STER_ENABLE_VERBOSITY "enable verbose config" )

# Build tests
option( L3STER_ENABLE_TESTS "enable tests" )

# Set up coverage gathering
option( L3STER_ENABLE_COVERAGE "enable coverage" )

# Build benchmarks
option( L3STER_ENABLE_BENCHMARKS "enable benchmarks" )

#######################################################################################################################

# Detect dependencies

# Utilities
include( "cmake/DefineTrilinosTarget.cmake" )
include( "cmake/ImportLibrary.cmake" )

# Import libraries
# Trilinos
define_trilinos_target( ${L3STER_ENABLE_VERBOSITY} Tpetra Belos MueLu )
list( APPEND L3STER_DEPENDENCY_LIST Trilinos )

# Eigen
find_package( Eigen3 3.3 REQUIRED )
list( APPEND L3STER_DEPENDENCY_LIST Eigen3::Eigen )

# Hwloc
importlibrary( hwloc.h "${L3STER_ENABLE_VERBOSITY}" )

# Metis
importlibrary( metis.h "${L3STER_ENABLE_VERBOSITY}" )

# TBB
find_package( TBB REQUIRED )
list( APPEND L3STER_DEPENDENCY_LIST TBB::tbb )

#######################################################################################################################

# Define the L3STER target

add_library( L3STER INTERFACE )
target_include_directories( L3STER INTERFACE include )
set_target_properties( L3STER PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_20 )
target_link_libraries( L3STER INTERFACE ${L3STER_DEPENDENCY_LIST} )

# Explicitly state L3STER sources - this is helpful to the IDE, but not required for a correct build
file( GLOB_RECURSE L3STER_SOURCES
      LIST_DIRECTORIES false
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      CONFIGURE_DEPENDS
      include/*
      )
target_sources( L3STER INTERFACE ${L3STER_SOURCES} )

#######################################################################################################################

# Tests

if ( L3STER_ENABLE_TESTS )
    add_subdirectory( tests )
endif ()

#######################################################################################################################

# Benchmarks

if ( L3STER_ENABLE_BENCHMARKS )
    add_subdirectory( benchmarks )
endif ()

#######################################################################################################################