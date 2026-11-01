CPMAddPackage(
    NAME benchmark
    GITHUB_REPOSITORY google/benchmark
    VERSION 1.9.5
    EXCLUDE_FROM_ALL YES
    OPTIONS
        "BENCHMARK_ENABLE_TESTING OFF"
        "BENCHMARK_ENABLE_WERROR OFF"
)

add_executable(FastNoiseBenchmark 
    "FastNoiseBenchmark.cpp"
)

target_link_libraries(FastNoiseBenchmark
    FastNoise
    benchmark::benchmark
)


add_executable(FastNoiseCpp11Test
    "FastNoiseCpp11Include.cpp"
)

set_target_properties(FastNoiseCpp11Test PROPERTIES CXX_STANDARD 11)

target_link_libraries(FastNoiseCpp11Test
    FastNoise
)

add_subdirectory(regression)

# Test-only SIMD node that reads library nodes' gradients, compiled with the same floating point settings as the library
if(FASTNOISE2_STRICT_FP)
    fastsimd_create_dispatch_library(FastNoiseGradientProbe SOURCES "damped/GradientProbe.inl")
else()
    fastsimd_create_dispatch_library(FastNoiseGradientProbe RELAXED SOURCES "damped/GradientProbe.inl")
endif()
target_link_libraries(FastNoiseGradientProbe PRIVATE FastNoise)
# The dispatch library build defines FASTSIMD_EXPORT, which would also turn FastNoise's imports into exports
target_compile_definitions(FastNoiseGradientProbe PRIVATE FASTNOISE_STATIC_LIB)
if(NOT FASTNOISE2_STRICT_FP)
    if(MSVC)
        target_compile_options(FastNoiseGradientProbe PRIVATE /fp:fast)
    else()
        target_compile_options(FastNoiseGradientProbe PRIVATE -ffast-math)
    endif()
endif()

add_executable(FastNoiseDampedTests "damped/FastNoiseDampedTests.cpp")
target_link_libraries(FastNoiseDampedTests PRIVATE FastNoise FastSIMD FastNoiseGradientProbe)

add_test(NAME FastNoiseDampedTests COMMAND FastNoiseDampedTests)
