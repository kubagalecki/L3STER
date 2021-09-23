# L3STER tests are divided into 2 categories, based on whether they require MPI. Non-MPI tests are based on the Catch2
#   testing framework and are compiled into a single executable. Individual tests are registered with CTest using the
#   `catch_discover_tests` utility provided by Catch2. Conversely, MPI tests are all compiled into separate executables.
#   These are defined and registered with CTest using the `add_mpi_test` utility (see below).

### Set up testing facilities ###

find_package( Catch2 REQUIRED )
include( CTest )
include( Catch )

#######################################################################################################################

### Define the Catch2-based target ###

add_executable( L3STER_tests TestsMain.cpp
                HwlocTests.cpp
                MappingTests.cpp
                MathTests.cpp
                MeshSerializationTests.cpp
                MeshTests.cpp
                NodeAllocationTests.cpp
                QuadratureTests.cpp
                UtilTests.cpp
                )
target_include_directories( L3STER_tests PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
target_link_libraries( L3STER_tests
                       L3STER
                       Catch2::Catch2
                       )

catch_discover_tests( L3STER_tests )

#######################################################################################################################

### Define MPI tests ###

# Utility for defining a test
function( add_mpi_test source name nprocs )
    get_filename_component( test_target ${source} NAME_WE )
    add_executable( ${test_target} ${source} )
    target_include_directories( ${test_target} PRIVATE ${CMAKE_CURRENT_BINARY_DIR} )
    target_compile_options( ${test_target} PRIVATE -fno-sanitize=all )
    target_link_options( ${test_target} PRIVATE -fno-sanitize=all )
    target_link_libraries( ${test_target} L3STER )
    set( test_env GMON_OUT_PREFIX=profile_data )
    set( test_driver mpiexec )
    if ( ${ARGC} EQUAL 4 )
        set( args ${ARGV4} )
    endif ()
    foreach ( np ${nprocs} )
        set( test_params "-n;${np};--oversubscribe;$<TARGET_FILE:${test_target}>;${args}" )
        set( test_name ${name}_np_${np} )
        set( test_dir ${CMAKE_CURRENT_BINARY_DIR}/${test_name} )
        file( MAKE_DIRECTORY ${test_dir} )
        list( APPEND L3STER_MPI_TESTS ${test_name} )
        add_test( NAME ${test_name}
                  COMMAND ${test_driver} ${test_params}
                  WORKING_DIRECTORY ${test_dir}
                  )
        set_tests_properties( ${test_name} PROPERTIES ENVIRONMENT ${test_env} )
    endforeach ()
    set( L3STER_MPI_TESTS ${L3STER_MPI_TESTS} PARENT_SCOPE )
endfunction()

# Define individual tests
add_mpi_test( "MpiBlockingCommunicationTest.cpp" "MPI_blocking_communication_test" "1;2;4" )
add_mpi_test( "MpiAsyncCommunicationTest.cpp" "MPI_async_communication_test" "1;2;4" )
add_mpi_test( "MeshSendRecvTest.cpp" "Mesh_send_recv_test" "2;4" )
add_mpi_test( "MeshDistributionTest.cpp" "Mesh_distribution_test" "1;2;4" )

#######################################################################################################################

### Generate test data ###

execute_process( COMMAND lscpu
                 COMMAND grep -i numa
                 COMMAND head -1
                 COMMAND grep -o "[0-9]*"
                 OUTPUT_VARIABLE L3STER_N_NUMA_NODES
                 )
set( L3STER_TESTS_DATA_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data" )
configure_file( TestDataPath.h.in TestDataPath.h )

#######################################################################################################################

### Coverage gathering ###

if ( L3STER_ENABLE_COVERAGE )
    set( script_file ${CMAKE_CURRENT_BINARY_DIR}/generate_coverage_report.sh )
    file( WRITE ${script_file}
          "gcovr -x -d --exclude-unreachable-branches --exclude-throw-branches -r ${L3STER_SOURCE_DIR}/.. -e ${L3STER_SOURCE_DIR}/tests -o coverage_report.xml ./\n"
          "rm gmon.out\n"
          )
    foreach ( test ${L3STER_MPI_TESTS} )
        file( APPEND ${script_file}
              "cd ${test}\n"
              "gcovr -x -d --exclude-unreachable-branches --exclude-throw-branches -r ${L3STER_SOURCE_DIR}/.. -e ${L3STER_SOURCE_DIR}/tests -o coverage_report.xml ./\n"
              "rm ./profile_data*\n"
              "cd ..\n"
              )
    endforeach ()
endif ()

#######################################################################################################################