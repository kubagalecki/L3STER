# ---  detect_TBB  ---
# Find the TBB library
#
function( detect_TBB Verbosity )
    if ( Verbosity )
        message( STATUS "Detecting TBB" )
    endif ()
    find_package( TBB )
    if ( Verbosity )
        if ( NOT TBB_FOUND )
            message( STATUS "Detecting TBB - not found" )
        else ()
            message( STATUS "Detecting TBB - found" )
        endif ()
    endif ()
    if ( NOT TBB_FOUND )
        message( SEND_ERROR "Could not find TBB using the CMake find_package utility. Please make sure it is "
                 "installed on your system and indicate its location by passing TBB_DIR to CMake." )
    endif ()
endfunction()

detect_TBB( ${L3STER_ENABLE_VERBOSITY} )
