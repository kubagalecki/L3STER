function( generate_coverage_script )
    set( script_file ${CMAKE_CURRENT_BINARY_DIR}/generate_coverage_report.sh )
    file( WRITE ${script_file}
          "gcovr -x -d -r ${L3STER_DIR} -e ${L3STER_DIR}/tests -o coverage_report.xml ./\n"
          "rm gmon.out\n"
          )
    foreach ( test ${L3STER_MPI_TESTS} )
        file( APPEND ${script_file}
              "cd ${test}\n"
              "gcovr -x -d -r ${L3STER_DIR} -e ${L3STER_DIR}/tests -o coverage_report.xml ./\n"
              "rm ./profile_data*\n"
              "cd ..\n"
              )
    endforeach ()
endfunction()

generate_coverage_script()