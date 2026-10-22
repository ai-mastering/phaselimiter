cmake_minimum_required(VERSION 3.23)
project(phaselimiter LANGUAGES C CXX)

if(CMAKE_SOURCE_DIR STREQUAL CMAKE_BINARY_DIR)
  message(FATAL_ERROR "Use an out-of-source build: cmake -S . -B build/release")
endif()

list(PREPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")
if(NOT DEFINED CMAKE_CXX_STANDARD)
  set(CMAKE_CXX_STANDARD 11)
endif()
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
# Keep the existing /MD and /MDd ABI on MSVC, including bundled libraries.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>DLL")

option(PHASELIMITER_SANITIZERS "Enable ASan/UBSan and libstdc++ assertions (GCC/Clang)" OFF)

option(SANDYBRIDGE_SUPPORT "Target AVX instead of AVX2/FMA (x86 only)" OFF)
option(PHASELIMITER_BUILD_BENCHMARKS "Build Google Benchmark executables" OFF)
option(PHASELIMITER_WARNINGS_AS_ERRORS "Treat project warnings as errors (GCC/Clang only)" ON)
option(PHASELIMITER_STATIC_BOOST "Prefer static Boost libraries" ${UNIX})

# Accept old command lines while keeping tests and benchmarks independent.
if(DEFINED DISABLE_TARGET_TEST)
  if(DISABLE_TARGET_TEST)
    set(BUILD_TESTING OFF CACHE BOOL "Build tests" FORCE)
  else()
    set(BUILD_TESTING ON CACHE BOOL "Build tests" FORCE)
  endif()
endif()
if(DEFINED DISABLE_TARGET_BENCH)
  if(DISABLE_TARGET_BENCH)
    set(PHASELIMITER_BUILD_BENCHMARKS OFF CACHE BOOL "Build benchmarks" FORCE)
  else()
    set(PHASELIMITER_BUILD_BENCHMARKS ON CACHE BOOL "Build benchmarks" FORCE)
  endif()
endif()
include(CTest)
include(GNUInstallDirs)
include(Dependencies)
include(ProjectOptions)
include(Cmaes)

add_library(bakuage STATIC
  deps/bakuage/src/bessel.cpp
  deps/bakuage/src/biquad_iir_filter.cpp
  deps/bakuage/src/convolution.cpp
  deps/bakuage/src/dbesi0.cpp
  deps/bakuage/src/dft.cpp
  deps/bakuage/src/dissonance.cpp
  deps/bakuage/src/file_utils.cpp
  deps/bakuage/src/fir_filter4.cpp
  deps/bakuage/src/get_peak_rss.cpp
  deps/bakuage/src/loudness_contours.cpp
  deps/bakuage/src/loudness_ebu_r128.cpp
  deps/bakuage/src/loudness_filter.cpp
  deps/bakuage/src/memory.cpp
  deps/bakuage/src/rnnoise.cpp
  deps/bakuage/src/sndfile_wrapper.cpp
  deps/bakuage/src/stacktrace.cpp
  deps/bakuage/src/utils.cpp
  deps/bakuage/src/vector_math.cpp
  deps/bakuage/src/window_func.cpp
)
target_include_directories(bakuage PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/bakuage/include")
target_link_libraries(bakuage PUBLIC phaselimiter_options phaselimiter_dependencies)
phaselimiter_enable_warnings(bakuage)

add_subdirectory(src)
