# Keep the encoding policy on executables: static libraries do not carry a
# process manifest. Include test/benchmark tools so their path behavior agrees.
function(phaselimiter_enable_windows_utf8 target)
  if(NOT WIN32)
    return()
  endif()
  set(manifest "${PROJECT_SOURCE_DIR}/packaging/windows/utf8.manifest")
  if(MSVC)
    # CMake passes .manifest sources to link.exe/lld-link and merges them with
    # the generated manifest (including the default requestedExecutionLevel).
    target_sources(${target} PRIVATE "${manifest}")
  else()
    # GNU-style Windows toolchains do not use CMake's MSVC manifest handling.
    # A resource also works with Clang/LLD cross builds on Linux.
    target_sources(${target} PRIVATE "${PROJECT_BINARY_DIR}/windows-utf8.rc")
  endif()
endfunction()

if(WIN32 AND NOT MSVC)
  enable_language(RC)
  set(PHASELIMITER_UTF8_MANIFEST "${PROJECT_SOURCE_DIR}/packaging/windows/utf8.manifest")
  configure_file("${PROJECT_SOURCE_DIR}/packaging/windows/utf8.rc.in"
    "${PROJECT_BINARY_DIR}/windows-utf8.rc" @ONLY)
  set_source_files_properties("${PROJECT_BINARY_DIR}/windows-utf8.rc" PROPERTIES
    OBJECT_DEPENDS "${PHASELIMITER_UTF8_MANIFEST}")
endif()
