# Standalone use builds the dump against another FastNoise2 checkout, such as the base commit:
#   cmake -S tests/regression -B out/build/regression-base -DFASTNOISE2_SOURCE=<base checkout>
set(regression_standalone OFF)
if(NOT DEFINED FastNoise2_SOURCE_DIR)
    set(regression_standalone ON)
    cmake_minimum_required(VERSION 3.10)
    project(FastNoiseRegression)
    set(CMAKE_CXX_STANDARD 17)
    set(FASTNOISE2_SOURCE "" CACHE PATH "FastNoise2 source tree to build the dump against")
    set(FASTNOISE2_TOOLS OFF CACHE BOOL "" FORCE)
    set(FASTNOISE2_TESTS OFF CACHE BOOL "" FORCE)
    set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
    add_subdirectory(${FASTNOISE2_SOURCE} FastNoise2)
endif()

add_executable(FastNoiseRegressionDump "FastNoiseRegressionDump.cpp")
target_link_libraries(FastNoiseRegressionDump PRIVATE FastNoise)

if(NOT regression_standalone)
    set(FASTNOISE2_REGRESSION_BASELINE "" CACHE PATH "Directory with metadata.txt and samples.bin dumped by the base build")
    set(fixtures "${CMAKE_CURRENT_SOURCE_DIR}/../fixtures/base-trees.txt")

    add_test(NAME RegressionDumpMetadata COMMAND FastNoiseRegressionDump metadata "${CMAKE_CURRENT_BINARY_DIR}/metadata.txt")
    add_test(NAME RegressionDumpSamples COMMAND FastNoiseRegressionDump samples "${fixtures}" "${CMAKE_CURRENT_BINARY_DIR}/samples.bin")

    if(FASTNOISE2_REGRESSION_BASELINE)
        add_test(NAME RegressionCompareSamples
            COMMAND ${CMAKE_COMMAND} -E compare_files "${FASTNOISE2_REGRESSION_BASELINE}/samples.bin" "${CMAKE_CURRENT_BINARY_DIR}/samples.bin")
        add_test(NAME RegressionCompareMetadata
            COMMAND ${CMAKE_COMMAND}
                -DBASE=${FASTNOISE2_REGRESSION_BASELINE}/metadata.txt
                -DFORK=${CMAKE_CURRENT_BINARY_DIR}/metadata.txt
                -P "${CMAKE_CURRENT_SOURCE_DIR}/CompareMetadata.cmake")
        set_tests_properties(RegressionCompareSamples PROPERTIES DEPENDS RegressionDumpSamples)
        set_tests_properties(RegressionCompareMetadata PROPERTIES DEPENDS RegressionDumpMetadata)
    endif()
endif()
