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
#   PackageNames    (list)   - semicolon-separated list of packages to find (case insensitive)
#
# Return values (i.e. variables set in parent scope):
#   MissingPackages (list)   - semicolon-separated list of packages which were not found
#
function( detect_trilinos_packages Verbosity PackageList PackageNames )

    if ( NOT PackageNames )
        return()
    endif ()

    if ( Verbosity )
        message( STATUS "Detecting required Trilinos packages" )
        list( APPEND CMAKE_MESSAGE_INDENT "  " )
    endif ()

    unset( MissingPackages )

    string( TOLOWER "${PackageList}" PackageList )

    foreach ( pkg IN LISTS PackageNames )
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
#   Verbosity               (bool)   - determines whether function will print status
#   PackageName...          (string) - each argument after the first will be treated as a case-
#                                        insensitive name of a Trilinos package to be found
#
# Return values (i.e. variables set in parent scope):
#   MissingTrilinosPackages (list)   - semicolon-separated list containing the names of the
#                                        packages which were requested but not built, set only if
#                                        at least one package was not found
#
function( define_trilinos_target Verbosity )

    if ( Verbosity )
        message( STATUS "Detecting Trilinos" )
        list( APPEND CMAKE_MESSAGE_INDENT "  " )
    endif ()

    find_package( Trilinos REQUIRED )

    if ( NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER} )
        message( WARNING " Detected different C++ compiler than the one Trilinos was built with.\n"
                 " Detected compiler:               ${CMAKE_CXX_COMPILER}\n"
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

    detect_trilinos_packages( ${Verbosity} "${Trilinos_PACKAGE_LIST}" "${ARGN}" )
    if ( MissingPackages )
        string( REPLACE ";" "\n > " fmt_mp "${MissingPackages}" )
        message( SEND_ERROR " Trilinos was built without the following required packages: \n > ${fmt_mp}\n" )
        set( MissingTrilinosPackages "${MissingPackages}" PARENT_SCOPE )
    endif ()

    if ( Verbosity )
        list( POP_BACK CMAKE_MESSAGE_INDENT )
        message( STATUS "Detecting Trilinos - found" )
    endif ()

    add_library( Trilinos INTERFACE )

    if ( Trilinos_CXX_COMPILER_FLAGS )
        string( STRIP ${Trilinos_CXX_COMPILER_FLAGS} Trilinos_CXX_COMPILER_FLAGS )
        string( REPLACE " " ";" Trilinos_CXX_COMPILER_FLAGS ${Trilinos_CXX_COMPILER_FLAGS} )
        target_compile_options( Trilinos INTERFACE ${Trilinos_CXX_COMPILER_FLAGS} )
    endif ()

    if ( Trilinos_BUILD_SHARED_LIBS AND Trilinos_SHARED_LIB_RPATH_COMMAND )
        string( STRIP ${Trilinos_SHARED_LIB_RPATH_COMMAND} Trilinos_SHARED_LIB_RPATH_COMMAND )
        string( REPLACE " " ";" Trilinos_SHARED_LIB_RPATH_COMMAND ${Trilinos_SHARED_LIB_RPATH_COMMAND} )
        target_link_options( Trilinos INTERFACE ${Trilinos_SHARED_LIB_RPATH_COMMAND} )
    endif ()

    if ( Trilinos_EXTRA_LD_FLAGS )
        string( STRIP ${Trilinos_EXTRA_LD_FLAGS} Trilinos_EXTRA_LD_FLAGS )
        string( REPLACE " " ";" Trilinos_EXTRA_LD_FLAGS ${Trilinos_EXTRA_LD_FLAGS} )
        target_link_options( Trilinos INTERFACE ${Trilinos_EXTRA_LD_FLAGS} )
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