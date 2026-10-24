# Honor both Unix and Windows conda layouts without global include/link paths.
foreach(prefix IN ITEMS "$ENV{CONDA_PREFIX}" "$ENV{CONDA_ROOT}")
  if(prefix)
    list(PREPEND CMAKE_PREFIX_PATH "${prefix}" "${prefix}/Library")
  endif()
endforeach()
set(Boost_USE_STATIC_LIBS ${PHASELIMITER_STATIC_BOOST})
find_package(Boost 1.65 CONFIG QUIET COMPONENTS filesystem serialization system math_tr1 iostreams)
if(NOT Boost_FOUND)
  # Older installations do not ship BoostConfig.cmake.
  if(POLICY CMP0167)
    cmake_policy(SET CMP0167 OLD)
  endif()
  find_package(Boost 1.65 REQUIRED COMPONENTS filesystem serialization system math_tr1 iostreams)
endif()
find_package(Threads REQUIRED)
find_package(PNG REQUIRED)
find_package(JPEG REQUIRED)
find_package(ZLIB REQUIRED)
# Apple Silicon uses the OS DSP library; x86 keeps the existing IPP backend.
if(APPLE AND PHASELIMITER_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(PHASELIMITER_USE_ACCELERATE ON)
  find_library(PHASELIMITER_ACCELERATE_FRAMEWORK Accelerate REQUIRED)
  set(phaselimiter_dsp_library "${PHASELIMITER_ACCELERATE_FRAMEWORK}")
else()
  set(PHASELIMITER_USE_ACCELERATE OFF)
  find_package(IPP REQUIRED)
  set(phaselimiter_dsp_library IPP::IPP)
endif()
find_package(TBB 2021.8 CONFIG REQUIRED COMPONENTS tbb tbbmalloc)
find_package(SndFile REQUIRED)
# optim uses Armadillo's header-only mode (ARMA_DONT_USE_WRAPPER).
find_path(ARMADILLO_INCLUDE_DIR NAMES armadillo REQUIRED)

add_library(phaselimiter_dependencies INTERFACE)
target_link_libraries(phaselimiter_dependencies INTERFACE
  Boost::filesystem Boost::serialization Boost::system Boost::math_tr1 Boost::iostreams
  ${phaselimiter_dsp_library} TBB::tbb TBB::tbbmalloc SndFile::SndFile PNG::PNG JPEG::JPEG ZLIB::ZLIB Threads::Threads
  ${CMAKE_DL_LIBS})
if(UNIX)
  target_link_libraries(phaselimiter_dependencies INTERFACE m)
elseif(WIN32)
  target_link_libraries(phaselimiter_dependencies INTERFACE shlwapi psapi)
endif()
target_include_directories(phaselimiter_dependencies SYSTEM INTERFACE
  "${ARMADILLO_INCLUDE_DIR}"
  "${PROJECT_SOURCE_DIR}/deps/eigen"
  "${PROJECT_SOURCE_DIR}/deps/CImg"
  "${PROJECT_SOURCE_DIR}/deps/libsimdpp"
  "${PROJECT_SOURCE_DIR}/deps/hnsw"
  "${PROJECT_SOURCE_DIR}/deps/picojson")

# Scope third-party options and policy compatibility to their subdirectories.
# CMake 4 removed compatibility with the old minimums in these pinned submodules.
function(phaselimiter_add_bundled_dependencies)
  if(CMAKE_VERSION VERSION_GREATER_EQUAL 4.0)
    set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
  endif()
  set(GFLAGS_BUILD_gflags_LIB ON)
  set(GFLAGS_BUILD_gflags_nothreads_LIB OFF)
  set(GFLAGS_BUILD_SHARED_LIBS OFF)
  set(GFLAGS_BUILD_STATIC_LIBS ON)
  set(GFLAGS_BUILD_TESTING OFF)
  # This old gflags revision also reads unprefixed parent options.
  set(project_build_testing "${BUILD_TESTING}")
  set(BUILD_TESTING OFF)
  set(BUILD_SHARED_LIBS OFF)
  set(BUILD_STATIC_LIBS ON)
  add_subdirectory(deps/gflags EXCLUDE_FROM_ALL)
  set(BUILD_TESTING "${project_build_testing}")
  if(BUILD_TESTING)
    # Suppress GoogleTest's legacy /MD-to-/MT flag rewrite; CMP0091 lets
    # CMAKE_MSVC_RUNTIME_LIBRARY select the CRT for all bundled targets.
    set(gtest_force_shared_crt ON CACHE BOOL "Use the shared MSVC runtime" FORCE)
    set(BUILD_GMOCK OFF CACHE BOOL "Build Google Mock" FORCE)
    set(INSTALL_GTEST OFF CACHE BOOL "Install GoogleTest" FORCE)
    add_subdirectory(deps/googletest EXCLUDE_FROM_ALL)
  endif()
endfunction()
phaselimiter_add_bundled_dependencies()

# Generate optim headers in the build tree using CMake, on every platform.
set(optim_generated_dir "${PROJECT_BINARY_DIR}/generated/optim")
file(GLOB_RECURSE optim_inputs CONFIGURE_DEPENDS
  "${PROJECT_SOURCE_DIR}/deps/optim/include/*"
  "${PROJECT_SOURCE_DIR}/deps/optim/src/*.cpp")
add_custom_command(
  OUTPUT "${optim_generated_dir}/generated.stamp"
  COMMAND "${CMAKE_COMMAND}"
    "-DOPTIM_SOURCE_DIR=${PROJECT_SOURCE_DIR}/deps/optim"
    "-DOPTIM_OUTPUT_DIR=${optim_generated_dir}"
    -P "${CMAKE_CURRENT_LIST_DIR}/GenerateOptim.cmake"
  COMMAND "${CMAKE_COMMAND}" -E touch "${optim_generated_dir}/generated.stamp"
  DEPENDS ${optim_inputs} "${CMAKE_CURRENT_LIST_DIR}/GenerateOptim.cmake"
  COMMENT "Generating optim header-only library" VERBATIM)
add_custom_target(phaselimiter_optim_headers DEPENDS "${optim_generated_dir}/generated.stamp")
add_library(phaselimiter_optim INTERFACE)
add_dependencies(phaselimiter_optim phaselimiter_optim_headers)
target_include_directories(phaselimiter_optim SYSTEM INTERFACE "${optim_generated_dir}")
