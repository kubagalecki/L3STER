# ---  detect_catch2  ---
# Detect whether Catch2 was provided as a directory, or whether L3STER should look for installation
# via find_package
#
function(detect_catch2 Verbosity L3STERPath)
    if (Verbosity)
        message(STATUS "Detecting Catch2")
        list(APPEND CMAKE_MESSAGE_INDENT "  ")
        message(STATUS "Looking for Catch2 in the L3STER directory")
    endif ()

    if (EXISTS "${L3STERPath}/Catch2")
        set(L3STER_Catch2_path "${L3STERPath}/Catch2" PARENT_SCOPE)
        if (Verbosity)
            message(STATUS "Looking for Catch2 in the L3STER directory - found")
            list(POP_BACK CMAKE_MESSAGE_INDENT)
            message(STATUS "Detecting Catch2 - found")
        endif ()
    endif ()
endfunction()

# Catch2 needs to be added at base scope to properly load the CTest registration module
if (L3STER_ENABLE_TESTS)
    detect_catch2(${L3STER_Verbosity} ${L3STER_DIR})

    if (L3STER_Catch2_path)
        add_subdirectory(${L3STER_Catch2_path})
    else ()
        if (L3STER_Verbosity)
            list(APPEND CMAKE_MESSAGE_INDENT "  ")
            message(STATUS "Looking for Catch2 in the L3STER directory - not found")
            message(STATUS "Checking if Catch2 was installed on the system")
        endif ()
        find_package(Catch2)
        if (TARGET Catch2::Catch2)
            if (L3STER_Verbosity)
                message(STATUS "Checking if Catch2 was installed on the system - found")
                list(POP_BACK CMAKE_MESSAGE_INDENT)
                message(STATUS "Detecting Catch2 - found")
            endif ()
        else ()
            if (L3STER_Verbosity)
                message(STATUS "Checking if Catch2 was installed on the system - not found")
                list(POP_BACK CMAKE_MESSAGE_INDENT)
                message(STATUS "Detecting Catch2 - not found")
            endif ()
            message(SEND_ERROR "Could not locate Catch2 unit test library. "
                    "If you don't have Catch2, you can simply clone the Catch2 repository "
                    "into the top level directory of L3STER.")
        endif ()
    endif ()

    include(CTest)
    include(Catch)
endif ()