# The fork must keep every base node ID and name, and add FractalDamped as the last ID
file(STRINGS "${BASE}" base_lines)
file(STRINGS "${FORK}" fork_lines)

list(GET base_lines 0 base_count_line)
list(GET fork_lines 0 fork_count_line)
string(REGEX REPLACE "count " "" base_count "${base_count_line}")
string(REGEX REPLACE "count " "" fork_count "${fork_count_line}")
math(EXPR expected_count "${base_count} + 1")

if(NOT fork_count EQUAL expected_count)
    message(FATAL_ERROR "Node count ${fork_count}, expected ${expected_count}")
endif()

foreach(i RANGE 1 ${base_count})
    list(GET base_lines ${i} base_line)
    list(GET fork_lines ${i} fork_line)
    if(NOT base_line STREQUAL fork_line)
        message(FATAL_ERROR "Node changed: base '${base_line}', fork '${fork_line}'")
    endif()
endforeach()

list(GET fork_lines ${expected_count} last_line)
if(NOT last_line STREQUAL "${base_count} FractalDamped")
    message(FATAL_ERROR "Last node is '${last_line}', expected '${base_count} FractalDamped'")
endif()

message(STATUS "${base_count} base nodes unchanged, FractalDamped is ID ${base_count}")
