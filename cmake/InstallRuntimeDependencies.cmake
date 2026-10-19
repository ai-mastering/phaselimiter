# Resolve actual DLL imports, including their transitive dependencies. Never
# enumerate DLL names or assume where conda was installed.
if(WIN32)
  set(PHASELIMITER_RUNTIME_DIRS "" CACHE STRING "Additional directories containing runtime DLLs")
  set(runtime_dirs ${PHASELIMITER_RUNTIME_DIRS}
    "$ENV{CONDA_PREFIX}/Library/bin"
    "$ENV{CONDA_ROOT}/Library/bin"
    "${PROJECT_SOURCE_DIR}/prebuilt/win64/libsndfile-1.2.2-win64/bin")
  foreach(prefix IN LISTS CMAKE_PREFIX_PATH)
    list(APPEND runtime_dirs "${prefix}/bin")
  endforeach()
  list(FILTER runtime_dirs EXCLUDE REGEX "^/Library/bin$")
  install(RUNTIME_DEPENDENCY_SET phaselimiter_runtime
    DIRECTORIES ${runtime_dirs}
    PRE_EXCLUDE_REGEXES "api-ms-.*" "ext-ms-.*"
    POST_EXCLUDE_REGEXES "^[A-Za-z]:[/\\\\][Ww][Ii][Nn][Dd][Oo][Ww][Ss][/\\\\].*"
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
elseif(UNIX AND NOT APPLE)
  # Bundle non-system dependencies such as legacy TBB. Keep glibc and the OS
  # media libraries with the package manager rather than shipping copies.
  install(RUNTIME_DEPENDENCY_SET phaselimiter_runtime
    PRE_EXCLUDE_REGEXES "linux-vdso.*" "^libstdc[+][+][.]so"
      "^libgcc_s[.]so" "^libc[.]so" "^ld-linux"
    POST_EXCLUDE_REGEXES "^/lib/" "^/lib64/" "^/usr/lib/" "^/usr/lib64/"
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR})
endif()
