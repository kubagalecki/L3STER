function( add_mpi_test source name nprocs )
    get_filename_component( test_target ${source} NAME_WE )
    add_executable( ${test_target} ${source} )
    target_link_libraries( ${test_target} L3STER )
    if ( ${ARGC} EQUAL 4 )
        set( args ${ARGV4} )
    endif ()
    foreach ( np ${nprocs} )
        set( test_params "-n;${np};$<TARGET_FILE:${test_target}>;${args}" )
        set( test_name ${name}_np_${np} )
        set( test_dir ${CMAKE_CURRENT_BINARY_DIR}/${test_name} )
        file( MAKE_DIRECTORY ${test_dir} )
        list( APPEND L3STER_MPI_TESTS ${test_name} )
        add_test( NAME ${test_name}
                  COMMAND mpiexec ${test_params}
                  WORKING_DIRECTORY ${test_dir}
                  )
    endforeach ()
    set( L3STER_MPI_TESTS ${L3STER_MPI_TESTS} PARENT_SCOPE )
endfunction()

add_mpi_test( "${L3STER_DIR}/tests/MpiCommTest.cpp" "MPI_initialization_test" "1;2" )