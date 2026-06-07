cmake_minimum_required(VERSION 3.24)

project(
    Neuronix
    VERSION 0.1.0
    DESCRIPTION "A modern C++20 neural network framework built from scratch"
    LANGUAGES CXX
)

option(NEURONIX_BUILD_TESTS      "Build Neuronix tests"      ON)
option(NEURONIX_BUILD_EXAMPLES   "Build Neuronix examples"   ON)
option(NEURONIX_BUILD_BENCHMARKS "Build Neuronix benchmarks" OFF)

# Global C++20 baseline so FetchContent'd deps (Catch2) compile with the same
# standard as the library. target_compile_features on neuronix covers the
# library and its consumers, but not third-party subdirectories.
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# MSVC does not update __cplusplus unless this flag is set, which breaks
# third-party headers (Catch2, etc.) that rely on it to detect C++17/20 support.
if(MSVC)
    add_compile_options(/Zc:__cplusplus)
endif()

add_library(neuronix
    src/neuronix.cpp
    src/core/matrix.cpp
)

add_library(Neuronix::neuronix ALIAS neuronix)

target_compile_features(neuronix PUBLIC cxx_std_20)

target_include_directories(neuronix
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
)

if(MSVC)
    target_compile_options(neuronix PRIVATE /W4 /permissive-)
else()
    target_compile_options(neuronix PRIVATE -Wall -Wextra -Wpedantic)
endif()

include(CTest)

if(NEURONIX_BUILD_TESTS AND BUILD_TESTING)
    include(FetchContent)

    FetchContent_Declare(
        Catch2
        URL https://github.com/catchorg/Catch2/archive/refs/tags/v3.7.1.tar.gz
    )

    FetchContent_MakeAvailable(Catch2)

    list(APPEND CMAKE_MODULE_PATH ${catch2_SOURCE_DIR}/extras)

    add_executable(neuronix_tests
        tests/smoke_test.cpp
        tests/matrix_tests.cpp
    )

    target_link_libraries(neuronix_tests
        PRIVATE
            Neuronix::neuronix
            Catch2::Catch2WithMain
    )

    if(MSVC)
        target_compile_options(neuronix_tests PRIVATE /W4 /permissive-)
    else()
        target_compile_options(neuronix_tests PRIVATE -Wall -Wextra -Wpedantic)
    endif()

    include(Catch)
    catch_discover_tests(neuronix_tests)
endif()

if(NEURONIX_BUILD_EXAMPLES)
    add_executable(neuronix_hello
        examples/hello_neuronix.cpp
    )

    target_link_libraries(neuronix_hello
        PRIVATE
            Neuronix::neuronix
    )
endif()

if(NEURONIX_BUILD_BENCHMARKS)
    add_executable(neuronix_benchmarks
        benchmarks/benchmark_placeholder.cpp
    )

    target_link_libraries(neuronix_benchmarks
        PRIVATE
            Neuronix::neuronix
    )
endif()
