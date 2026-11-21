cmake_minimum_required(VERSION 3.16)
project(
    test_shared_ptr
    VERSION 1.0.0
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Suppress GCC's hardware interference size warning
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    add_compile_options(-Wno-interference-size)
endif()

# Direct binary outputs to build/bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

# Target definition
add_executable(
    test_shared_ptr
    test/test_shared_ptr.cpp
    test/test_shared_ptr.h
    shared_ptr/atomic_shared_ptr.h
    shared_ptr/rundown_shared_ptr.h
    shared_ptr/shared_ptr_utils.h
)

# Include directories
target_include_directories(
    test_shared_ptr
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/shared_ptr
        ${CMAKE_CURRENT_SOURCE_DIR}/test
)

# Concurrency / Threading primitives
find_package(Threads REQUIRED)
target_link_libraries(
    test_shared_ptr
    PRIVATE
        Threads::Threads
)

# 1. Fix Clang 16-byte atomic linker errors on Linux
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    target_link_libraries(
        test_shared_ptr
        PRIVATE
            atomic
    )
endif()

# NUMA Support (Linux only)[cite: 1]
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    find_library(NUMA_LIBRARY numa REQUIRED)
    target_link_libraries(test_shared_ptr PRIVATE ${NUMA_LIBRARY})
    message(STATUS "NUMA library found and linked: ${NUMA_LIBRARY}")
endif()

# 2. Force true lock-free hardware instructions for 128-bit atomics (e.g., cmpxchg16b)
include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-mcx16" HAS_MCX16)
if(HAS_MCX16)
    target_compile_options(
        test_shared_ptr
        PRIVATE
            -mcx16
    )
endif()

# 3. Maximum speed optimization without vectorization
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU")
    target_compile_options(
        test_shared_ptr
        PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-fomit-frame-pointer>
            $<$<CONFIG:Release>:-fno-tree-vectorize>
            $<$<CONFIG:Release>:-fno-tree-slp-vectorize>
    )
elseif(CMAKE_CXX_COMPILER_ID MATCHES "Clang|AppleClang")
    target_compile_options(
        test_shared_ptr
        PRIVATE
            $<$<CONFIG:Release>:-O3>
            $<$<CONFIG:Release>:-fomit-frame-pointer>
            $<$<CONFIG:Release>:-fno-vectorize>
            $<$<CONFIG:Release>:-fno-slp-vectorize>
    )
endif()

# Interprocedural / Link-Time Optimization (LTO)
include(CheckIPOSupported)
check_ipo_supported(
    RESULT ipo_supported
    OUTPUT ipo_error
)
if(ipo_supported)
    set_property(
        TARGET test_shared_ptr
        PROPERTY INTERPROCEDURAL_OPTIMIZATION_RELEASE TRUE
    )
endif()
