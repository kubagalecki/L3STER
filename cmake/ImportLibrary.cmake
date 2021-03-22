# --- importLibrary ---
# Locates the requested library on disc and wraps it into an imported target. This utility is meant for importing
# CMake-unaware libraries into CMake projects.
# Arguments:
#   1) name of the library header: `lib_name.hpp` (please include the extension)
#   2) boolean indicating whether to report progress
# Effects:
#   Creates an imported target named `lib_name`, which can be consumed as a dependency using target_link_libraries.
#   If the requested header or corresponding library cannot be found, a non-fatal error is reported.
#
function( importLibrary header verbosity )
    get_filename_component( lib_name ${header} NAME_WE )
    if ( verbosity )
        message( STATUS "Detecting ${lib_name}" )
    endif ()
    if ( ${${lib_name}_DIR} )
        find_path( lib_include_dir ${header} PATHS ${${lib_name}_DIR} PATH_SUFFIXES include NO_DEFAULT_PATH )
    endif ()
    find_path( lib_include_dir ${header} )
    if ( NOT lib_include_dir )
        if ( Verbosity )
            message( STATUS "Detecting ${lib_name} - not found" )
        endif ()
        message( SEND_ERROR "Could not locate ${header}. Try setting the variable ${lib_name}_DIR to indicate the "
                 "install path" )
        return()
    endif ()
    find_library( lib_path ${lib_name} HINTS ${lib_include_dir} ${lib_include_dir}/.. PATH_SUFFIXES lib )
    if ( NOT lib_path )
        if ( verbosity )
            message( STATUS "Detecting ${lib_name} - not found" )
        endif ()
        message( SEND_ERROR "Could not locate the ${lib_name} library in the expected directory relative to the header "
                 "file. Try setting the variable ${lib_name}_DIR to indicate the correct install path" )
        return()
    endif ()
    if ( verbosity )
        message( STATUS "Detecting ${lib_name} - found" )
    endif ()
    add_library( ${lib_name} UNKNOWN IMPORTED )
    target_include_directories( ${lib_name} INTERFACE ${lib_include_dir} )
    set_target_properties( ${lib_name} PROPERTIES IMPORTED_LOCATION ${lib_path} )
endfunction()