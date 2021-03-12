function( generate_coverage_script )
    set( script_file ${CMAKE_CURRENT_BINARY_DIR}/generate_coverage_report.sh )
    file( WRITE ${script_file}
          "gcovr --json -r ${L3STER_DIR} -e ${L3STER_DIR}/tests -o unit_tests_report.json\n"
          )
    foreach ( test ${L3STER_MPI_TESTS} )
        file( APPEND ${script_file}
              "cd ${test}\n"
              "gcovr --json -r ${L3STER_DIR} -e ${L3STER_DIR}/tests -o ${test}_report.json\n"
              "cd ..\n"
              )
    endforeach ()
    file( APPEND ${script_file} "gcovr -a ${CMAKE_CURRENT_BINARY_DIR}/unit_tests_report.json \\\n" )
    foreach ( test ${L3STER_MPI_TESTS} )
        file( APPEND ${script_file} "-a ${CMAKE_CURRENT_BINARY_DIR}/${test}/${test}_report.json \\\n" )
    endforeach ()
    file( APPEND ${script_file} "--json -r ${L3STER_DIR} -o coverage_report.json" )
endfunction()

generate_coverage_script()