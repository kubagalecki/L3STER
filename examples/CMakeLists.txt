# Note: this file is only intended to be used by the CI infrastructure, to make sure all examples build correctly
# If you are a human reader, please consider the individual examples, including their respective CMakeLists.txt files

cmake_minimum_required( VERSION 3.24 )
project( L3STER-examples-all )
add_subdirectory( .. L3STER-bin )
file( GLOB subdirs ${CMAKE_CURRENT_SOURCE_DIR}/* )
foreach ( dir IN LISTS subdirs )
    if ( IS_DIRECTORY "${dir}" )
        cmake_path( GET dir STEM name )
        if ( "${name}" STREQUAL build )
            continue()
        endif ()
        message( STATUS "Found example: ${name}" )
        file( GLOB src "${dir}/*.cpp" )
        add_executable( ${name} ${src} )
        target_link_libraries( ${name} L3STER )
    endif ()
endforeach ()