cmake_minimum_required(VERSION 3.18.1)
project("llama_jni")

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Native source directory is the current directory
set(NATIVE_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}")

# ============================================
# CRITICAL PERFORMANCE SETTINGS
# ============================================

# Configure llama.cpp for Android cross-compilation
set(LLAMA_NATIVE OFF CACHE BOOL "" FORCE)  # Critical for cross-compilation
set(LLAMA_STATIC ON CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)

# ENABLE FLASH ATTENTION - Major speedup!
set(LLAMA_FLASH_ATTN ON CACHE BOOL "" FORCE)

# Configure GGML for Android
set(GGML_NATIVE OFF CACHE BOOL "" FORCE)  # Critical for cross-compilation
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)  # OpenMP can cause issues on Android
set(GGML_ACCELERATE OFF CACHE BOOL "" FORCE)  # Apple-only
set(GGML_METAL OFF CACHE BOOL "" FORCE)  # Apple-only
set(GGML_CUDA OFF CACHE BOOL "" FORCE)  # Desktop GPU
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)  # Not needed for now
set(GGML_SYCL OFF CACHE BOOL "" FORCE)  # Intel-only
set(GGML_KOMPUTE OFF CACHE BOOL "" FORCE)  # Not needed
set(GGML_LLAMAFILE OFF CACHE BOOL "" FORCE)  # Disable - FP16 NEON code breaks on armeabi-v7a

# Enable CPU backend for ARM
set(GGML_CPU ON CACHE BOOL "" FORCE)

# Note: LTO disabled - causes linker issues on Windows NDK
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION ON)

# Add llama.cpp as subdirectory
add_subdirectory(${NATIVE_SRC_DIR}/llama.cpp ${CMAKE_BINARY_DIR}/llama_cpp)

# Include directories
include_directories(
    ${NATIVE_SRC_DIR}/llama.cpp/include
    ${NATIVE_SRC_DIR}/llama.cpp/ggml/include
    ${NATIVE_SRC_DIR}/llama.cpp/common
    ${NATIVE_SRC_DIR}
)

# Source files for our JNI wrapper
add_library(
    llama_jni
    SHARED
    ${NATIVE_SRC_DIR}/llama_jni.cpp
    ${NATIVE_SRC_DIR}/inference_engine.cpp
    ${NATIVE_SRC_DIR}/memory_manager.cpp
    ${NATIVE_SRC_DIR}/kv_cache.cpp
    ${NATIVE_SRC_DIR}/platform_channel.cpp
)

# Find required Android libraries
find_library(log-lib log)
find_library(android-lib android)

# Link libraries
target_link_libraries(
    llama_jni
    llama
    ggml
    ggml-cpu
    ggml-base
    ${log-lib}
    ${android-lib}
)

# Compiler flags for optimization - enable ARM NEON SIMD
target_compile_options(llama_jni PRIVATE
    -O3
    -ffast-math
    -fno-finite-math-only
    -fno-rtti
    -ftree-vectorize
    -fno-exceptions
    -funroll-loops
    -fomit-frame-pointer
)

# Architecture-specific NEON optimization flags
if(CMAKE_ANDROID_ARCH_ABI STREQUAL "arm64-v8a")
    # 64-bit ARM with full NEON SIMD + FP16 + dotprod support
    target_compile_options(llama_jni PRIVATE
        -march=armv8.2-a+fp16+dotprod+simd
        -mtune=cortex-a76
    )
    target_compile_definitions(llama_jni PRIVATE
        GGML_USE_NEON=1
        __ARM_NEON=1
        __ARM_FEATURE_FP16_VECTOR_ARITHMETIC=1
        __ARM_FEATURE_DOTPROD=1
    )
    message(STATUS "ARM64 optimizations: NEON + FP16 + DOTPROD + Flash Attention")
elseif(CMAKE_ANDROID_ARCH_ABI STREQUAL "armeabi-v7a")
    # 32-bit ARM with NEON
    target_compile_options(llama_jni PRIVATE
        -mfpu=neon-vfpv4
        -mfloat-abi=softfp
    )
    target_compile_definitions(llama_jni PRIVATE
        GGML_USE_NEON=1
        __ARM_NEON=1
    )
    message(STATUS "ARM32 optimizations: NEON")
endif()

# Enable Flash Attention define
target_compile_definitions(llama_jni PRIVATE
    GGML_USE_FLASH_ATTN=1
)
