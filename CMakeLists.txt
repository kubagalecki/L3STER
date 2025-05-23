cmake_minimum_required( VERSION 3.24 )

#######################################################################################################################

### Define project ###

project( L3STER
         LANGUAGES CXX
         VERSION 0.1.0
         )

#######################################################################################################################

### Require Linux ###

if ( NOT ${CMAKE_SYSTEM_NAME} STREQUAL Linux )
    message( FATAL_ERROR "Currently, L3STER only supports Linux" )
endif ()

#######################################################################################################################

### L3STER options ###

# Build tests
option( L3STER_ENABLE_TESTS "enable tests" )

# Set up coverage gathering
option( L3STER_ENABLE_COVERAGE "enable coverage" )

# Build benchmarks
option( L3STER_ENABLE_BENCHMARKS "enable benchmarks" )

#######################################################################################################################

### Detect dependencies ###

# Utilities
include( "cmake/DefineTrilinosTarget.cmake" )
include( "cmake/ImportLibrary.cmake" )

# Import libraries
# Trilinos
set( L3STER_REQUIRED_Trilinos_PACKAGES "Kokkos;Tpetra" )
set( L3STER_OPTIONAL_Trilinos_PACKAGES "Belos;Ifpack2;Amesos2;MueLu" )
find_trilinos( 14.0 "${L3STER_REQUIRED_Trilinos_PACKAGES}" "${L3STER_OPTIONAL_Trilinos_PACKAGES}" )
list( APPEND L3STER_DEPENDENCIES Trilinos::all_libs )

# Eigen
find_package( Eigen3 3.4 REQUIRED )
list( APPEND L3STER_DEPENDENCIES Eigen3::Eigen )

# TBB
find_package( TBB REQUIRED )
list( APPEND L3STER_DEPENDENCIES TBB::tbb )

# Hwloc
importlibrary( hwloc REQUIRED )
list( APPEND L3STER_IMPORTED_DEPENDENCIES hwloc-import )
list( APPEND L3STER_DEPENDENCIES hwloc-import )

# Metis
importlibrary( metis REQUIRED )
list( APPEND L3STER_IMPORTED_DEPENDENCIES metis-import )
list( APPEND L3STER_DEPENDENCIES metis-import )

#######################################################################################################################

### Define the L3STER target ###

include( "cmake/HardwareInfo.cmake" )
hardwareInfo()
list( JOIN L3STER_CACHE_SIZES "uz,\n  " L3STER_CACHE_SIZES )
string( APPEND L3STER_CACHE_SIZES "uz" )
configure_file( "${CMAKE_CURRENT_SOURCE_DIR}/include/l3ster/util/CacheSizesAtCompileTime.hpp.in"
                "${CMAKE_CURRENT_SOURCE_DIR}/include/l3ster/util/CacheSizesAtCompileTime.hpp"
                )

add_library( L3STER INTERFACE )
target_include_directories( L3STER INTERFACE
                            $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
                            $<INSTALL_INTERFACE:$<INSTALL_PREFIX>/include>
                            )
set_target_properties( L3STER PROPERTIES INTERFACE_COMPILE_FEATURES cxx_std_23 )
target_link_libraries( L3STER INTERFACE ${L3STER_DEPENDENCIES} )
foreach ( pkg IN LISTS L3STER_OPTIONAL_Trilinos_PACKAGES )
    string( TOUPPER "${pkg}" PKG )
    target_compile_definitions( L3STER INTERFACE "L3STER_TRILINOS_HAS_${PKG}" )
endforeach ()

# Explicitly state L3STER sources - this is helpful to the IDE, but not required for a correct build
file( GLOB_RECURSE L3STER_SOURCES
      LIST_DIRECTORIES false
      RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
      CONFIGURE_DEPENDS
      include/l3ster/*
      )
target_sources( L3STER INTERFACE $<BUILD_INTERFACE:${L3STER_SOURCES}> )

#######################################################################################################################

### Tests ###

if ( L3STER_ENABLE_TESTS )
    enable_testing()
    add_subdirectory( tests )
endif ()

#######################################################################################################################

### Benchmarks ###

if ( L3STER_ENABLE_BENCHMARKS )
    enable_testing()
    add_subdirectory( benchmarks )
endif ()

#######################################################################################################################

### Installation ###

include( GNUInstallDirs )
include( CMakePackageConfigHelpers )

# Mark L3STER for export
install( TARGETS L3STER ${L3STER_IMPORTED_DEPENDENCIES} EXPORT L3STERTargets )

# Export L3STER
install( EXPORT L3STERTargets
         DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/L3STER"
         NAMESPACE L3STER::
         )

# Copy L3STER headers to the destination directory
install( DIRECTORY include/l3ster
         DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
         FILES_MATCHING REGEX ".*\.h(pp)?"
         )

# Configure package and install config
configure_package_config_file( "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Config.cmake.in"
                               "${CMAKE_CURRENT_BINARY_DIR}/L3STERConfig.cmake"
                               INSTALL_DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/L3STER"
                               )
write_basic_package_version_file(
        "${CMAKE_CURRENT_BINARY_DIR}/L3STERConfigVersion.cmake"
        VERSION "${CMAKE_PROJECT_VERSION}"
        COMPATIBILITY AnyNewerVersion
        )
install( FILES
         "${CMAKE_CURRENT_BINARY_DIR}/L3STERConfig.cmake"
         "${CMAKE_CURRENT_BINARY_DIR}/L3STERConfigVersion.cmake"
         DESTINATION "${CMAKE_INSTALL_LIBDIR}/cmake/L3STER"
         )
