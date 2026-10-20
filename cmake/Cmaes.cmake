# Build the pinned libcmaes core with the same Eigen and compiler options as
# its callers. Avoid upstream's system Eigen lookup and optional dependencies.
set(cmaes_source "${PROJECT_SOURCE_DIR}/deps/libcmaes")
set(cmaes_generated "${PROJECT_BINARY_DIR}/generated/cmaes")
find_package(OpenMP REQUIRED COMPONENTS CXX)
file(MAKE_DIRECTORY "${cmaes_generated}/libcmaes")
include("${CMAKE_CURRENT_LIST_DIR}/GenerateCmaesHeader.cmake")
phaselimiter_generate_cmaes_header(
  "${cmaes_source}/include/libcmaes/eigenmvn.h"
  "${cmaes_generated}/libcmaes/eigenmvn.h")
file(CONFIGURE OUTPUT "${cmaes_generated}/libcmaes/libcmaes_config.h"
  CONTENT "// Optional logging and surrogate models are disabled.\n")
file(CONFIGURE OUTPUT "${cmaes_generated}/libcmaes/cmaes_export.h"
  CONTENT "#pragma once\n#define CMAES_EXPORT\n#define CMAES_NO_EXPORT\n")
add_library(phaselimiter_cmaes STATIC
  "${cmaes_source}/src/acovarianceupdate.cc"
  "${cmaes_source}/src/cmaparameters.cc"
  "${cmaes_source}/src/cmastopcriteria.cc"
  "${cmaes_source}/src/covarianceupdate.cc"
  "${cmaes_source}/src/esostrategy.cc"
  "${cmaes_source}/src/pwq_bound_strategy.cc"
  "${cmaes_source}/src/vdcmaupdate.cc"
  "${cmaes_source}/src/bipopcmastrategy.cc"
  "${cmaes_source}/src/cmasolutions.cc"
  "${cmaes_source}/src/cmastrategy.cc"
  "${cmaes_source}/src/errstats.cc"
  "${cmaes_source}/src/ipopcmastrategy.cc")
target_include_directories(phaselimiter_cmaes SYSTEM PUBLIC
  "${cmaes_generated}" "${cmaes_source}/include" "${PROJECT_SOURCE_DIR}/deps/eigen")
target_link_libraries(phaselimiter_cmaes PRIVATE phaselimiter_options)
target_link_libraries(phaselimiter_cmaes PUBLIC OpenMP::OpenMP_CXX)
if(MSVC)
  target_compile_options(phaselimiter_cmaes PRIVATE /bigobj)
endif()
install(FILES "${PROJECT_SOURCE_DIR}/packaging/licenses/libcmaes.txt"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/phaselimiter/licenses")
