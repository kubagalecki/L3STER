function( hardwareInfo )
    if ( L3STER_CACHE_SIZES )
        return()
    endif ()
    execute_process( COMMAND hwloc-info --version
                     RESULT_VARIABLE _hwloc_exit
                     OUTPUT_VARIABLE _
                     )
    if ( NOT _hwloc_exit EQUAL 0 )
        message( WARNING "Failed to detect CPU cache sizes: the hwloc-info utility is not available.\n"
                 "L3STER will assume default values, which may negatively affect performance.\n"
                 "You may explicitly specify cache sizes (in bytes) as a semicolon-separated list in L3STER_CACHE_SIZES"
                 )
        set( L3STER_CACHE_SIZES "32768;262144" PARENT_SCOPE )
        return()
    endif ()
    set( L3STER_CACHE_SIZES "" )
    execute_process( COMMAND hwloc-info --disallowed --silent core:all
                     COMMAND wc -l
                     OUTPUT_VARIABLE _num_cores
                     )
    foreach ( cache_lvl RANGE 1 5 )
        set( _sz 0 )
        execute_process( COMMAND hwloc-info --disallowed --silent "l${cache_lvl}cache:all"
                         COMMAND wc -l
                         OUTPUT_VARIABLE _num_caches
                         )
        if ( _num_caches EQUAL 0 )
            continue()
        endif ()
        math( EXPR _loop_last "${_num_caches} - 1" )
        foreach ( i RANGE "${_loop_last}" )
            execute_process( COMMAND hwloc-info --disallowed "l${cache_lvl}cache:${i}"
                             COMMAND grep "attr cache size"
                             COMMAND grep -o "[0-9]*"
                             OUTPUT_VARIABLE _isz
                             )
            math( EXPR _sz "${_sz} + ${_isz}" )
        endforeach ()
        math( EXPR _sz "${_sz} / ${_num_cores}" )
        list( APPEND L3STER_CACHE_SIZES "${_sz}" )
    endforeach ()
    list( LENGTH L3STER_CACHE_SIZES _num_caches )
    if ( _num_caches EQUAL 0 )
        message( WARNING "Failed to detect CPU cache sizes: the hwloc-info utility yielded no useful information.\n"
                 "L3STER will assume default values, which may negatively affect performance.\n"
                 "You may explicitly specify cache sizes (in bytes) as a semicolon-separated list in L3STER_CACHE_SIZES"
                 )
        set( L3STER_CACHE_SIZES "32768;262144" )
    endif ()
    set( L3STER_CACHE_SIZES "${L3STER_CACHE_SIZES}" PARENT_SCOPE )
endfunction()