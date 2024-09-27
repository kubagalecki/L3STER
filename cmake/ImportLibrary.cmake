# --- importLibrary ---
# Locates the requested library on disc and wraps it into an imported target. This utility is meant for importing
# CMake-unaware libraries into CMake projects.
# Arguments:
#   lib_name          (string)    -  name of the library
#   REQUIRED          (optional)  -  whether to signal an error if library isn't found
# Effects:
#   Creates an imported target named `lib_name::lib_name`, which can be consumed as a dependency using target_link_libraries.
#
function( importLibrary lib_name )
    message( STATUS "Detecting ${lib_name}" )

    list( FIND ARGV "REQUIRED" required_pos )
    if ( NOT ${required_pos} EQUAL -1 )
        set( required "TRUE" )
    endif ()

    unset( ${lib_name}_LIB CACHE )
    find_library( ${lib_name}_LIB ${lib_name} NO_CACHE )
    if ( ${lib_name}_LIB-NOTFOUND )
        message( STATUS "Detecting ${lib_name} - not found" )
        if ( required )
            message( FATAL_ERROR "The library \"${lib_name}\" was not found. Try setting ${lib_name}_ROOT=/path/to/library" )
        else ()
            set( ${lib_name}_FOUND "FALSE" PARENT_SCOPE )
            return()
        endif ()
    else ()
        message( STATUS "Detecting ${lib_name} - found: ${${lib_name}_LIB}" )
    endif ()
    cmake_path( GET ${lib_name}_LIB PARENT_PATH ${lib_name}_LIBDIR )

    add_library( ${lib_name}::${lib_name} IMPORTED UNKNOWN GLOBAL )
    set_target_properties( ${lib_name}::${lib_name} PROPERTIES IMPORTED_LOCATION "${${lib_name}_LIB}" )

    cmake_path( SET ${lib_name}_INCLUDEDIR NORMALIZE "${${lib_name}_LIBDIR}/../include" )
    cmake_path( ABSOLUTE_PATH ${lib_name}_INCLUDEDIR )
    if ( EXISTS "${${lib_name}_INCLUDEDIR}" AND IS_DIRECTORY "${${lib_name}_INCLUDEDIR}" )
        target_include_directories( ${lib_name}::${lib_name} INTERFACE "${${lib_name}_INCLUDEDIR}" )
    endif ()
endfunction()
