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
function(detect_trilinos_packages Verbosity PackageList PackageNames)

    if (NOT PackageNames)
        return()
    endif()

    if(Verbosity)
        message(STATUS "Detecting required Trilinos packages")
        list(APPEND CMAKE_MESSAGE_INDENT "  ")
    endif()

    unset(MissingPackages)

    string(TOLOWER "${PackageList}" PackageList)

    foreach(pkg IN LISTS PackageNames)
        if (Verbosity)
            message(STATUS "Detecting ${pkg}")
        endif()
        string(TOLOWER ${pkg} Name_LC)
        list(FIND PackageList ${Name_LC} pkg_index)
        if(NOT pkg_index EQUAL -1)
            if(Verbosity)
                message(STATUS "Detecting ${pkg} - found")
            endif()
        else()
            list(APPEND MissingPackages ${pkg})
            if(Verbosity)
                message(STATUS "Detecting ${pkg} - not found")
            endif()
        endif()
    endforeach()

    list(POP_BACK CMAKE_MESSAGE_INDENT)
    if (MissingPackages)
        string(REPLACE ";" "\n > " fmt_mp "${MissingPackages}")
        if (Verbosity)
            message(STATUS "Detecting required Trilinos packages - some not found")
        endif()
        message(SEND_ERROR " Trilinos was built without the following required packages: \n > ${fmt_mp}\n")
    else()
        if (Verbosity)
            message(STATUS "Detecting required Trilinos packages - all found")
        endif()
    endif()

    set(MissingPackages "${MissingPackages}" PARENT_SCOPE)
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
#                                        packages which were requested but not built
function(make_trilinos_target Verbosity)

    find_package(Trilinos REQUIRED)

    if (NOT ${Trilinos_CXX_COMPILER} STREQUAL ${CMAKE_CXX_COMPILER})
        message(WARNING " Detected different C++ compiler than the one Trilinos was built with.\n"
            " Detected compiler:               ${CMAKE_CXX_COMPILER}\n"
            " Compiler used to build Trilinos: ${Trilinos_CXX_COMPILER}\n"
            "Note: if the difference is e.g. `cxx` vs `mpicxx`, you can ignore this warning. "
            "Trilinos includes MPI in the libraries it links against, so your application will link against the same MPI by virtue of the transitive property.\n")
    endif()

    detect_trilinos_packages(${Verbosity} "${Trilinos_PACKAGE_LIST}" "${ARGN}")
    set(MissingTrilinosPackages "${MissingPackages}" PARENT_SCOPE)

    add_library(Trilinos INTERFACE)

    string(STRIP ${Trilinos_CXX_COMPILER_FLAGS} Trilinos_CXX_COMPILER_FLAGS)
    string(REPLACE " " ";" Trilinos_CXX_COMPILER_FLAGS ${Trilinos_CXX_COMPILER_FLAGS})
    target_compile_options(Trilinos INTERFACE ${Trilinos_CXX_COMPILER_FLAGS})

    string(STRIP ${Trilinos_EXTRA_LD_FLAGS} Trilinos_EXTRA_LD_FLAGS)
    string(REPLACE " " ";" Trilinos_EXTRA_LD_FLAGS ${Trilinos_EXTRA_LD_FLAGS})
    target_link_options(Trilinos INTERFACE ${Trilinos_EXTRA_LD_FLAGS})

    target_include_directories(Trilinos INTERFACE
        ${Trilinos_INCLUDE_DIRS}
        ${Trilinos_TPL_INCLUDE_DIRS}
    )

    target_link_directories(Trilinos INTERFACE
        ${Trilinos_LIBRARY_DIRS}
        ${Trilinos_TPL_LIBRARY_DIRS}
    )

    target_link_libraries(Trilinos INTERFACE
        ${Trilinos_LIBRARIES}
        ${Trilinos_TPL_LIBRARIES}
    )
endfunction()
