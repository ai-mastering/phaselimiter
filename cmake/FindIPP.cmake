# Legacy IPP and oneAPI installations have different library layouts.
set(IPPROOT "$ENV{IPPROOT}" CACHE PATH "Intel IPP installation prefix")
set(_ipp_hints "${IPPROOT}" /opt/intel/oneapi/ipp/latest /opt/intel/ipp)
find_path(IPP_INCLUDE_DIR NAMES ipp.h HINTS ${_ipp_hints} PATH_SUFFIXES include)
set(_ipp_required IPP_INCLUDE_DIR)
foreach(component IN ITEMS ippi ipps ippvm ippcore)
  find_library(IPP_${component}_LIBRARY NAMES ${component} ${component}mt
    HINTS ${_ipp_hints} PATH_SUFFIXES lib lib/intel64 lib/intel64_win)
  list(APPEND _ipp_required IPP_${component}_LIBRARY)
endforeach()
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(IPP REQUIRED_VARS ${_ipp_required})
if(IPP_FOUND AND NOT TARGET IPP::IPP)
  add_library(IPP::IPP INTERFACE IMPORTED)
  set_target_properties(IPP::IPP PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${IPP_INCLUDE_DIR}"
    INTERFACE_LINK_LIBRARIES "${IPP_ippi_LIBRARY};${IPP_ipps_LIBRARY};${IPP_ippvm_LIBRARY};${IPP_ippcore_LIBRARY}")
endif()
mark_as_advanced(${_ipp_required})
