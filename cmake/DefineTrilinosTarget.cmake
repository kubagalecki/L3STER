# ---  detect_trilinos_packages  ---
# Convenience function for checking Trilinos packages
#
# This function checks that the passed Trilinos packages have been built. If any of them cannot be
#   found, it will produce a non-fatal error (via `message(SEND_ERROR ...)`). If you wish to
#   terminate the config process in this event, you need to manually check the MissingPackages
#   variable after this function completes.
#
# Arguments:
#   Verbosity       (bool)   - determines whether function will print status
#   PackageList     (string) - built Trilinos packages (Trilinos_PACKAGE_LIST variable)
#   PackageNamesReq (list)   - semicolon-separated list of required Trilinos packages (case insensitive)
#   PackageNamesOpt (list)   - semicolon-separated list of optional Trilinos packages (case insensitive), optional argument
#
# Return values (i.e. variables set in parent scope):
#   MissingPackages (list)          - semicolon-separated list of required packages which were not found
#   Trilinos_<Package>_FOUND (bool) - flags indicating which optional packages were found
#
function( detect_trilinos_packages Verbosity PackageList PackageNamesReq )

    string( TOLOWER "${PackageList}" PackageList )
    unset( MissingPackages )
    if ( Verbosity )
        message( STATUS "Detecting required Trilinos packages" )
        list( APPEND CMAKE_MESSAGE_INDENT "  " )
    endif ()
    foreach ( pkg IN LISTS PackageNamesReq )
        if ( Verbosity )
            message( STATUS "Detecting ${pkg}" )
        endif ()
        string( TOLOWER ${pkg} Name_LC )
        list( FIND PackageList ${Name_LC} pkg_index )
        if ( NOT pkg_index EQUAL -1 )
            if ( Verbosity )
                message( STATUS "Detecting ${pkg} - found" )
            endif ()
        else ()
            list( APPEND MissingPackages ${pkg} )
            if ( Verbosity )
                message( STATUS "Detecting ${pkg} - not found" )
            endif ()
        endif ()
    endforeach ()

    if ( Verbosity )
        list( POP_BACK CMAKE_MESSAGE_INDENT )
        if ( MissingPackages )
            message( STATUS "Detecting required Trilinos packages - some not found" )
        else ()
            message( STATUS "Detecting required Trilinos packages - all found" )
        endif ()
    endif ()
    set( MissingPackages "${MissingPackages}" PARENT_SCOPE )

    if ( ${ARGC} GREATER_EQUAL 4 )
        set( PackageNamesOpt "${ARGV3}" )
        if ( Verbosity )
            message( STATUS "Detecting optional Trilinos packages" )
            list( APPEND CMAKE_MESSAGE_INDENT "  " )
        endif ()
        foreach ( pkg IN LISTS PackageNamesOpt )
            if ( Verbosity )
                message( STATUS "Detecting ${pkg}" )
            endif ()
            string( TOLOWER ${pkg} Name_LC )
            list( FIND PackageList ${Name_LC} pkg_index )
            if ( NOT pkg_index EQUAL -1 )
                if ( Verbosity )
                    message( STATUS "Detecting ${pkg} - found" )
                endif ()
                set( Trilinos_${pkg}_FOUND ON PARENT_SCOPE )
            else ()
                if ( Verbosity )
                    message( STATUS "Detecting ${pkg} - not found" )
                endif ()
            endif ()
        endforeach ()
        if ( Verbosity )
            list( POP_BACK CMAKE_MESSAGE_INDENT )
            message( STATUS "Detecting optional Trilinos packages - finished" )
        endif ()
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

# ---  make_trilinos_target  ---
# Convert variables exported by the `find_package(Trilinos)` call into a linkable CMake target
#
# This function will create an interface target (via `add_library(Trilinos INTERFACE)`) which can
#   be linked against in the usual `target_link_libraries` way. The target sets the following
#   interface properties, based on the results of the call to `find_package(Trilinos)`:
#     - C++ compiler flags
#     - C++ linker flags
#     - Include directories (including TPLs)
#     - Link directories (including TPLs)
#     - Linkage against static/shared libraries (including TPLs)
#   The modification of this script to include C and Fortran compiler and linker flags is
#   straightforward. For a better understanding of the inner workings of this script, the user
#   can refer to [https://trilinos.github.io/pdfs/Finding_Trilinos.txt], which documents all
#   variables set by calling `find_package(Trilinos)` and constitutes the basis for this script.
#   Additionaly, this function can detect whether specific Trilinos packages were built, and error
#   out if they were not. To provide flexibility, the error is non-fatal. If the user wishes to
#   terminate the config process in this event, they can check the MissingTrilinosPackages
#   variable (see below) after this function completes.
#
# Arguments:
#   Verbosity        (bool)    - determines whether function will print status
#   Version          (string)  - minimum required version of trilinos
#   PackageNamesReq  (list)    - semicolon-separated list of Trilinos packages which are required, optional argument
#   PackageNamesOpt  (list)    - semicolon-separated list of Trilinos packages which are optional, optional argument
#
# Return values (i.e. variables set in parent scope):
#   MissingTrilinosPackages (list)   - semicolon-separated list containing the names of the
#                                      packages which were requested but not built, set only if
#                                      at least one package was not found
#
function( define_trilinos_target Verbosity Version )

    if ( Verbosity )
        message( STATUS "Detecting Trilinos" )
        list( APPEND CMAKE_MESSAGE_INDENT "  " )
    endif ()

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

    if ( ${ARGC} GREATER_EQUAL 4 )
        detect_trilinos_packages( ${Verbosity} "${Trilinos_PACKAGE_LIST}" "${ARGV2}" "${ARGV3}" )
        foreach ( pkg IN LISTS PackageNamesOpt )
            if ( Trilinos_${pkg}_FOUND )
                set( Trilinos_${pkg}_FOUND ON PARENT_SCOPE )
            endif ()
        endforeach ()
    elseif ( ${ARGC} EQUAL 3 )
        detect_trilinos_packages( ${Verbosity} "${Trilinos_PACKAGE_LIST}" "${ARGV2}" )
    elseif ( ${ARGC} EQUAL 2 )
        detect_trilinos_packages( ${Verbosity} "${Trilinos_PACKAGE_LIST}" "" )
    endif ()
    if ( MissingPackages )
        string( REPLACE ";" "\n- " fmt_mp "${MissingPackages}" )
        message( SEND_ERROR " Trilinos was built without the following required packages:\n- ${fmt_mp}\n" )
        set( MissingTrilinosPackages "${MissingPackages}" PARENT_SCOPE )
    endif ()

    if ( Verbosity )
        list( POP_BACK CMAKE_MESSAGE_INDENT )
        message( STATUS "Detecting Trilinos - found" )
    endif ()

    add_library( Trilinos INTERFACE )

    if ( Trilinos_CXX_COMPILER_FLAGS )
        string( STRIP ${Trilinos_CXX_COMPILER_FLAGS} Trilinos_CXX_COMPILER_FLAGS )
        string( REPLACE " " ";" Trilinos_CXX_COMPILER_FLAGS "${Trilinos_CXX_COMPILER_FLAGS}" )
        if ( Verbosity )
            message( STATUS "Compiler flags set by Trilinos: ${Trilinos_CXX_COMPILER_FLAGS}" )
        endif ()
        target_compile_options( Trilinos INTERFACE ${Trilinos_CXX_COMPILER_FLAGS} )
        string( FIND "${Trilinos_CXX_COMPILER_FLAGS}" "-fopenmp" FOPENMP_FLAG_POS )
        if ( NOT ${FOPENMP_FLAG_POS} EQUAL -1 )
            set( Trilinos_HAS_OPENMP ON )
        endif ()
    endif ()
    if ( Trilinos_HAS_OPENMP )
        find_package( OpenMP )
        if ( OpenMP_CXX_FOUND )
            string( REPLACE " " ";" OpenMP_CXX_FLAGS "${OpenMP_CXX_FLAGS}" )
            target_compile_options( Trilinos INTERFACE "${OpenMP_CXX_FLAGS}" )
            target_include_directories( Trilinos INTERFACE "${OpenMP_CXX_INCLUDE_DIRS}" )
            target_link_libraries( Trilinos INTERFACE "${OpenMP_CXX_LIBRARIES}" )
        else ()
            message( SEND_ERROR "Trilinos was built with OpenMP, but the current compiler lacks OpenMP support for C++" )
        endif ()
    endif ()

    if ( Trilinos_BUILD_SHARED_LIBS AND Trilinos_SHARED_LIB_RPATH_COMMAND )
        string( STRIP ${Trilinos_SHARED_LIB_RPATH_COMMAND} Trilinos_SHARED_LIB_RPATH_COMMAND )
        string( REPLACE " " ";" Trilinos_SHARED_LIB_RPATH_COMMAND "${Trilinos_SHARED_LIB_RPATH_COMMAND}" )
        target_link_options( Trilinos INTERFACE ${Trilinos_SHARED_LIB_RPATH_COMMAND} )
    endif ()

    if ( Trilinos_EXTRA_LD_FLAGS )
        string( STRIP ${Trilinos_EXTRA_LD_FLAGS} Trilinos_EXTRA_LD_FLAGS )
        string( REPLACE " " ";" Trilinos_EXTRA_LD_FLAGS "${Trilinos_EXTRA_LD_FLAGS}" )
        target_link_options( Trilinos INTERFACE ${Trilinos_EXTRA_LD_FLAGS} )
        if ( Verbosity )
            message( STATUS "Linker flags set by Trilinos: ${Trilinos_EXTRA_LD_FLAGS}" )
        endif ()
    endif ()

    target_include_directories( Trilinos INTERFACE
                                ${Trilinos_INCLUDE_DIRS}
                                ${Trilinos_TPL_INCLUDE_DIRS}
                                )

    target_link_directories( Trilinos INTERFACE
                             ${Trilinos_LIBRARY_DIRS}
                             ${Trilinos_TPL_LIBRARY_DIRS}
                             )

    target_link_libraries( Trilinos INTERFACE
                           ${Trilinos_LIBRARIES}
                           ${Trilinos_TPL_LIBRARIES}
                           )
endfunction()