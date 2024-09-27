# ---  detect_trilinos_packages  ---
# Convenience function for checking Trilinos packages
#
# This function checks that the passed Trilinos packages have been built. If any of them cannot be
#   found, it will produce a non-fatal error (via `message(SEND_ERROR ...)`). If you wish to
#   terminate the config process in this event, you need to manually check the MissingPackages
#   variable after this function completes.
#
# Arguments:
#   PackageList     (string) - built Trilinos packages (Trilinos_PACKAGE_LIST variable)
#   PackageNamesReq (list)   - semicolon-separated list of required Trilinos packages (case insensitive)
#   PackageNamesOpt (list)   - semicolon-separated list of optional Trilinos packages (case insensitive), optional argument
#
# Return values (i.e. variables set in parent scope):
#   MissingPackages (list)          - semicolon-separated list of required packages which were not found
#   Trilinos_<Package>_FOUND (bool) - flags indicating which optional packages were found
#
function( detect_trilinos_packages PackageList PackageNamesReq )

    string( TOLOWER "${PackageList}" PackageList )
    unset( MissingPackages )
    message( STATUS "Detecting required Trilinos packages" )
    list( APPEND CMAKE_MESSAGE_INDENT "  " )
    foreach ( pkg IN LISTS PackageNamesReq )
        message( STATUS "Detecting ${pkg}" )
        string( TOLOWER ${pkg} Name_LC )
        list( FIND PackageList ${Name_LC} pkg_index )
        if ( NOT pkg_index EQUAL -1 )
            message( STATUS "Detecting ${pkg} - found" )
        else ()
            list( APPEND MissingPackages ${pkg} )
            message( STATUS "Detecting ${pkg} - not found" )
        endif ()
    endforeach ()

    list( POP_BACK CMAKE_MESSAGE_INDENT )
    if ( MissingPackages )
        message( STATUS "Detecting required Trilinos packages - some not found" )
    else ()
        message( STATUS "Detecting required Trilinos packages - all found" )
    endif ()
    set( MissingPackages "${MissingPackages}" PARENT_SCOPE )

    if ( ${ARGC} GREATER_EQUAL 3 )
        set( PackageNamesOpt "${ARGV2}" )
        message( STATUS "Detecting optional Trilinos packages" )
        list( APPEND CMAKE_MESSAGE_INDENT "  " )
        foreach ( pkg IN LISTS PackageNamesOpt )
            message( STATUS "Detecting ${pkg}" )
            string( TOLOWER ${pkg} Name_LC )
            list( FIND PackageList ${Name_LC} pkg_index )
            if ( NOT pkg_index EQUAL -1 )
                message( STATUS "Detecting ${pkg} - found" )
                set( Trilinos_${pkg}_FOUND ON PARENT_SCOPE )
            else ()
                message( STATUS "Detecting ${pkg} - not found" )
            endif ()
        endforeach ()
        list( POP_BACK CMAKE_MESSAGE_INDENT )
        message( STATUS "Detecting optional Trilinos packages - finished" )
    endif ()

endfunction()

###################################################################################################

# --- check_package_version ---
# CMake package version requirements are incomprehensible, so this is a simple hand-rolled utility
# which compares the versions in lexicographical order, e.g., 1.1 >= 1.0 >= 0.9 >= 0.8.1 >= 0.8.1
# No input validation is performed.

function( check_package_version Name Required Provided )
    string( COMPARE EQUAL ${Provided} "" ProvidedEmpty )
    if ( ProvidedEmpty )
        set( "${Name}_VERSION_OK" OFF PARENT_SCOPE )
        return()
    endif ()

    string( REPLACE "." ";" RequiredParsed "${Required}" )
    string( REPLACE "." ";" ProvidedParsed "${Provided}" )
    while ( 1 )
        list( LENGTH RequiredParsed ReqLen )
        list( LENGTH ProvidedParsed ProvLen )
        if (( ReqLen EQUAL 0 ) OR ( ProvLen EQUAL 0 ))
            break()
        endif ()
        list( GET RequiredParsed 0 ReqFront )
        list( GET ProvidedParsed 0 ProvFront )
        if ( NOT ReqFront EQUAL ProvFront )
            break()
        endif ()
        list( POP_BACK RequiredParsed )
        list( POP_BACK ProvidedParsed )
    endwhile ()

    list( LENGTH RequiredParsed ReqLen )
    list( LENGTH ProvidedParsed ProvLen )
    if (( ReqLen EQUAL 0 ) OR ( ProvLen EQUAL 0 ))
        if ( ProvLen GREATER_EQUAL ReqLen )
            set( ${Name}_VERSION_OK ON PARENT_SCOPE )
        else ()
            set( ${Name}_VERSION_OK OFF PARENT_SCOPE )
        endif ()
    else ()
        list( GET RequiredParsed 0 ReqFront )
        list( GET ProvidedParsed 0 ProvFront )
        if ( ProvFront GREATER_EQUAL ReqFront )
            set( ${Name}_VERSION_OK ON PARENT_SCOPE )
        else ()
            set( ${Name}_VERSION_OK OFF PARENT_SCOPE )
        endif ()
    endif ()
endfunction()

###################################################################################################

# ---  find_trilinos  ---
# Arguments:
#   Version          (string)  - minimum required version of trilinos
#   PackageNamesReq  (list)    - semicolon-separated list of Trilinos packages which are required, optional argument
#   PackageNamesOpt  (list)    - semicolon-separated list of Trilinos packages which are optional, optional argument
#
# Return values (i.e. variables set in parent scope):
#   MissingTrilinosPackages (list)   - semicolon-separated list containing the names of the
#                                      packages which were requested but not built, set only if
#                                      at least one package was not found
#
function( find_trilinos Version )

    message( STATUS "Detecting Trilinos" )
    list( APPEND CMAKE_MESSAGE_INDENT "  " )

    find_package( Trilinos REQUIRED )
    check_package_version( Trilinos "${Version}" "${Trilinos_VERSION}" )
    if ( NOT Trilinos_VERSION_OK )
        message( SEND_ERROR "The provided version of Trilinos is less than the required version.\nProvided: ${Trilinos_VERSION}. Required: ${Version}" )
    endif ()

    file( REAL_PATH "${Trilinos_CXX_COMPILER}" Trilinos_CXX_COMPILER )
    file( REAL_PATH "${CMAKE_CXX_COMPILER}" abspath_CXX_COMPILER )
    if ( NOT "${Trilinos_CXX_COMPILER}" STREQUAL "${abspath_CXX_COMPILER}" )
        message( WARNING " Detected different C++ compiler than the one Trilinos was built with.\n"
                 " Detected compiler:               ${abspath_CXX_COMPILER}\n"
                 " Compiler used to build Trilinos: ${Trilinos_CXX_COMPILER}\n"
                 "You should likely be using an MPI compiler wrapper (e.g. mpic++) to compile L3STER applications. "
                 "The wrapper is responsible for linking against MPI. If you have multiple versions of MPI installed, "
                 "please make sure you are using the same one which was used to build Trilinos. Otherwise, you may get "
                 "linker errors, or worse: hard to detect runtime breaks. This warning is here to ensure you are aware "
                 "that you are responsible for ensuring compatibility. If you're uncertain of what this all means, "
                 "it's probably safest to force CMake to configure using the compiler specified above by passing:\n"
                 " -DCMAKE_CXX_COMPILER=${Trilinos_CXX_COMPILER}\n"
                 "or setting the equivalent in a toolchain file.\n" )
    endif ()

    if ( ${ARGC} GREATER_EQUAL 3 )
        detect_trilinos_packages( "${Trilinos_PACKAGE_LIST}" "${ARGV1}" "${ARGV2}" )
        foreach ( pkg IN LISTS PackageNamesOpt )
            if ( Trilinos_${pkg}_FOUND )
                set( Trilinos_${pkg}_FOUND ON PARENT_SCOPE )
            endif ()
        endforeach ()
    elseif ( ${ARGC} EQUAL 2 )
        detect_trilinos_packages( ${Verbosity} "${Trilinos_PACKAGE_LIST}" "${ARGV1}" )
    elseif ( ${ARGC} EQUAL 1 )
        detect_trilinos_packages( ${Verbosity} "${Trilinos_PACKAGE_LIST}" "" )
    endif ()
    if ( MissingPackages )
        string( REPLACE ";" "\n- " fmt_mp "${MissingPackages}" )
        message( SEND_ERROR " Trilinos was built without the following required packages:\n- ${fmt_mp}\n" )
        set( MissingTrilinosPackages "${MissingPackages}" PARENT_SCOPE )
    endif ()

    list( POP_BACK CMAKE_MESSAGE_INDENT )
    message( STATUS "Detecting Trilinos - found" )

endfunction()