add_library(phaselimiter_options INTERFACE)
target_compile_features(phaselimiter_options INTERFACE cxx_std_11)
target_compile_definitions(phaselimiter_options INTERFACE
  ARMA_DONT_USE_HDF5 EIGEN_DONT_PARALLELIZE OPTIM_USE_TBB cimg_display=0 cimg_use_png cimg_use_jpeg)

if(PHASELIMITER_TARGET_PROCESSOR MATCHES "^(x86_64|AMD64|amd64|i.86|x86|X86)$")
  set(phaselimiter_x86 ON)
  target_compile_definitions(phaselimiter_options INTERFACE SIMDPP_ARCH_X86_AVX)
  if(NOT SANDYBRIDGE_SUPPORT)
    target_compile_definitions(phaselimiter_options INTERFACE
      SIMDPP_ARCH_X86_AVX2 SIMDPP_ARCH_X86_FMA3 BA_FMA_ENABLED)
  endif()
elseif(APPLE AND PHASELIMITER_TARGET_PROCESSOR MATCHES "^(arm64|aarch64)$")
  set(phaselimiter_x86 OFF)
  target_compile_definitions(phaselimiter_options INTERFACE SIMDPP_ARCH_ARM_NEON)
else()
  message(FATAL_ERROR "Supported targets: x86 with IPP, or macOS ARM64 with Accelerate")
endif()
if(PHASELIMITER_USE_ACCELERATE)
  target_compile_definitions(phaselimiter_options INTERFACE BAKUAGE_USE_ACCELERATE)
endif()
if(MSVC)
  # Windows.h must not include legacy WinSock.h before Boost.Asio uses WinSock2.
  target_compile_definitions(phaselimiter_options INTERFACE NOMINMAX WIN32_LEAN_AND_MEAN)
  target_compile_options(phaselimiter_options INTERFACE /MP /utf-8)
  if(SANDYBRIDGE_SUPPORT)
    target_compile_options(phaselimiter_options INTERFACE /arch:AVX)
  else()
    target_compile_options(phaselimiter_options INTERFACE /arch:AVX2)
  endif()
elseif(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(NOT phaselimiter_x86)
    # ARM64 includes NEON; do not pass x86 ISA options to AppleClang.
  elseif(SANDYBRIDGE_SUPPORT)
    target_compile_options(phaselimiter_options INTERFACE -march=sandybridge -mtune=sandybridge)
  else()
    target_compile_options(phaselimiter_options INTERFACE -march=core-avx2 -mtune=core-avx2)
  endif()
else()
  message(FATAL_ERROR "Supported compilers: GCC, Clang, AppleClang, MSVC")
endif()

function(phaselimiter_enable_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W3)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wno-sign-compare
      -Wno-unused-variable -Wno-unused-but-set-variable)
    if(PHASELIMITER_WARNINGS_AS_ERRORS)
      target_compile_options(${target} PRIVATE -Werror)
    endif()
  endif()
endfunction()

# Apply to project code, including bakuage, through its public options target.
# Keep release artifacts and third-party dependency build flags independent.
if(PHASELIMITER_SANITIZERS)
  if(MSVC OR NOT CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    message(FATAL_ERROR "PHASELIMITER_SANITIZERS requires GCC or Clang")
  endif()
  target_compile_options(phaselimiter_options INTERFACE
    -O1 -g -fno-omit-frame-pointer -fsanitize=address,undefined
    -fno-sanitize-recover=all)
  # GCC's stricter array checks complement ASan and the default UBSan bounds checks.
  if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(phaselimiter_options INTERFACE -fsanitize=bounds-strict)
  endif()
  target_link_options(phaselimiter_options INTERFACE -fsanitize=address,undefined)
  # Effective when using libstdc++; harmless when using another C++ library.
  target_compile_definitions(phaselimiter_options INTERFACE _GLIBCXX_ASSERTIONS)
endif()
