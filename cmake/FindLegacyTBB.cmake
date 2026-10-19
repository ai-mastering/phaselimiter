# The source uses TBB 2019/2020 APIs removed in oneTBB 2021.
set(TBB_ROOT "$ENV{TBB_ROOT}" CACHE PATH "Legacy Intel TBB installation prefix (2019/2020)")
find_package(TBB CONFIG QUIET COMPONENTS tbb tbbmalloc)
if(TARGET TBB::tbb)
  get_target_property(_tbb_includes TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
endif()
find_path(LegacyTBB_INCLUDE_DIR NAMES tbb/task_scheduler_init.h
  HINTS "${TBB_ROOT}" ${_tbb_includes} PATH_SUFFIXES include)
set(_tbb_required LegacyTBB_INCLUDE_DIR)
if(NOT TARGET TBB::tbb OR NOT TARGET TBB::tbbmalloc)
  foreach(component IN ITEMS tbb tbbmalloc)
    find_library(LegacyTBB_${component}_LIBRARY NAMES ${component}
      HINTS "${TBB_ROOT}" PATH_SUFFIXES lib lib/intel64/gcc4.8)
    list(APPEND _tbb_required LegacyTBB_${component}_LIBRARY)
  endforeach()
endif()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LegacyTBB REQUIRED_VARS ${_tbb_required}
  REASON_FAILURE_MESSAGE "Install TBB 2019/2020 and set TBB_ROOT or CMAKE_PREFIX_PATH. oneTBB 2021+ requires source changes.")
if(LegacyTBB_FOUND AND NOT TARGET LegacyTBB::LegacyTBB)
  # Reject a modern TBB config paired accidentally with legacy headers.
  if(DEFINED TBB_VERSION AND TBB_VERSION VERSION_GREATER_EQUAL 2021)
    message(FATAL_ERROR "oneTBB ${TBB_VERSION} is unsupported; point TBB_DIR at a TBB 2019/2020 installation.")
  endif()
  add_library(LegacyTBB::LegacyTBB INTERFACE IMPORTED)
  if(TARGET TBB::tbb AND TARGET TBB::tbbmalloc)
    set(_tbb_libraries TBB::tbb TBB::tbbmalloc)
  else()
    set(_tbb_libraries "${LegacyTBB_tbb_LIBRARY}" "${LegacyTBB_tbbmalloc_LIBRARY}")
  endif()
  set_target_properties(LegacyTBB::LegacyTBB PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${LegacyTBB_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${_tbb_libraries}")
endif()
mark_as_advanced(${_tbb_required})
