cmake_minimum_required(VERSION 3.9)

project(phaselimiter C CXX)
include(ExternalProject)

# set(CMAKE_VERBOSE_MAKEFILE ON)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)

# enable link time optimization
# https://stackoverflow.com/questions/31355692/cmake-support-for-gccs-link-time-optimization-lto
# 逆に遅くなる
# set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)

# windowsの場合は、debugとreleaseでlibディレクトリを分ける
# includeを使う
# linuxの場合はそのままシステムから探させる

# Visual Studio 11 2012 Win64
# Visual Studio 11 2012    <-- Win32 (なんかWin64だとうまくいかないからこっちを使う)

# for sandy bridge
option(SANDYBRIDGE_SUPPORT "support sandy bridge (linux/apple only)" OFF)
if(SANDYBRIDGE_SUPPORT)
  add_definitions(-DSIMDPP_ARCH_X86_AVX)
else()
  add_definitions(-DSIMDPP_ARCH_X86_AVX)
  add_definitions(-DSIMDPP_ARCH_X86_AVX2)
  add_definitions(-DSIMDPP_ARCH_X86_FMA3)
  add_compile_definitions(BA_FMA_ENABLED)
endif()

option(DISABLE_BENCH "disable bench" OFF)

find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# arma_H5Ovisitが見つからないエラー回避 https://stackoverflow.com/questions/62590370/c-armadillo-procedure-entry-point-h5ovisit-could-not-be-located-in-libarmadi
# add_definitions(-DARMA_DONT_USE_HDF5)

# add_definitions(-DPHASELIMITER_ENABLE_FFTW)

if(DEFINED ENV{IPPROOT})
    set(IPPROOT $ENV{IPPROOT})
else()
    set(IPPROOT /opt/intel/ipp)
endif()
message(${IPPROOT})

add_compile_definitions(cimg_display=0)
add_definitions(-DEIGEN_DONT_PARALLELIZE)
add_definitions(-DOPTIM_ENABLE_EIGEN_WRAPPERS)
add_definitions(-DOPTIM_USE_OMP)
if(WIN32)
else()
  add_definitions(-Dcimg_use_png)
endif()

if(WIN32)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} /MP /utf-8 /arch:AVX")
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} /MD")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} /MDd")
  if(CMAKE_SIZEOF_VOID_P EQUAL 8)
    set(BAKUAGE_PREBUILT_DIR ${PROJECT_SOURCE_DIR}/prebuilt/win64)
    set(BAKUAGE_LIB_DIR ${BAKUAGE_PREBUILT_DIR}/lib)
    include_directories(
      prebuilt/win64/libsndfile-1.2.2-win64/include
    )
    link_directories(
      prebuilt/win64/libsndfile-1.2.2-win64/lib
    )
    set(Boost_ARCHITECTURE "-x64")
  elseif(CMAKE_SIZEOF_VOID_P EQUAL 4)
    set(BAKUAGE_PREBUILT_DIR ${PROJECT_SOURCE_DIR}/prebuilt/win32)
    set(BAKUAGE_LIB_DIR ${BAKUAGE_PREBUILT_DIR}/lib/release)
  endif()

  set(Boost_USE_STATIC_LIBS ON)
  set(Boost_USE_MULTITHREADED ON)
  set(Boost_USE_STATIC_RUNTIME ON)
  # set(Boost_COMPILER "-vc142")
  set(Boost_DEBUG OFF)
  find_package(Boost REQUIRED system filesystem serialization math_tr1 iostreams)

  set(BAKUAGE_LINK_LIBRARIES
    # ${BAKUAGE_LIB_DIR}/libfftw3-3.lib
    # ${BAKUAGE_LIB_DIR}/libfftw3f-3.lib
    # ${BAKUAGE_LIB_DIR}/libFLAC_dynamic.lib
    sndfile.lib
    # ${BAKUAGE_LIB_DIR}/gflags_static.lib
    shlwapi.lib
    ippimt.lib
    ippsmt.lib
    ippcoremt.lib
    ippvmmt.lib
    ${Boost_LIBRARIES}
  )
  link_directories(
    ${Boost_LIBRARY_DIRS}
  	${IPPROOT}/lib
  )
  include_directories(
  	${BAKUAGE_PREBUILT_DIR}/include
  	${Boost_INCLUDE_DIRS}
  	${IPPROOT}/include
  )
  set(TINY_PROCESS_SOURCES
    deps/tiny-process-library/process.hpp
    deps/tiny-process-library/process.cpp
    deps/tiny-process-library/process_win.cpp
  )
  set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
elseif(APPLE)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
  if(SANDYBRIDGE_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sandybridge -mtune=sandybridge -mavx")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2 -mtune=core-avx2 -mavx2 -mfma")
  endif()
  SET(CMAKE_EXE_LINKER_FLAGS  "${CMAKE_EXE_LINKER_FLAGS} -rpath /opt/intel/ipp/lib")

  find_package(Boost REQUIRED COMPONENTS filesystem math_tr1 serialization system)

  set(BAKUAGE_LIB_DIR ${BAKUAGE_PREBUILT_DIR}/lib)
  set(BAKUAGE_LINK_LIBRARIES
     # libfftw3.a
     # libfftw3f.a
     # libgflags.dylib
     libsndfile.a
     # libFLAC.a
     # libvorbisenc.a
     # libvorbis.a
     # libogg.a
     libboost_system.a
     libboost_filesystem.a
     libboost_serialization.a
     libboost_math_tr1.a
     libboost_iostreams.a
     libtbb.a
     libtbbmalloc.a
     /opt/intel/ipp/lib/libippi.dylib
     /opt/intel/ipp/lib/libipps.dylib
     /opt/intel/ipp/lib/libippvm.dylib
     /opt/intel/ipp/lib/libippcore.dylib
     libpng.a
     libz.a
     pthread
     m
  )
  link_directories(
    ${Boost_LIBRARY_DIRS}
    ${IPPROOT}/lib
    /opt/intel/tbb/lib
    /usr/local/opt/libtensorflow/lib
  )
  include_directories(
    ${Boost_INCLUDE_DIRS}
    ${IPPROOT}/include
    /opt/intel/tbb/include
  )
  set(TINY_PROCESS_SOURCES
    deps/tiny-process-library/process.hpp
    deps/tiny-process-library/process.cpp
    deps/tiny-process-library/process_unix.cpp
  )
else()
  # -fopt-info-vec-optimized-missed
  # auto vectorizeはそれほど速くならない。(理由: auto vectorizeできるようなところはそもそも単純で遅くない？)
  # 出力が冗長になるのでオフにした
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -frecord-gcc-switches -Werror -Wall -Wno-sign-compare -Wno-unused-variable -Wno-unused-but-set-variable -Wno-enum-compare")
  if(SANDYBRIDGE_SUPPORT)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=sandybridge -mtune=sandybridge -mavx")
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=core-avx2 -mtune=core-avx2 -mavx2 -mfma")
  endif()

  # https://otland.net/threads/problem-with-build-on-ubuntu-pie-object.260285/
  set(CMAKE_CXX_LINK_FLAGS "${CMAKE_CXX_LINK_FLAGS} -no-pie")

  set(BAKUAGE_PREBUILT_DIR ${PROJECT_SOURCE_DIR}/prebuilt/linux)
  set(BAKUAGE_LIB_DIR ${BAKUAGE_PREBUILT_DIR}/lib)
  set(BAKUAGE_LINK_LIBRARIES
   	 # libfftw3.a
   	 # libfftw3f.a
     # libgflags.a
     libsndfile.so
     # libFLAC.a
     # libvorbisenc.a
     # libvorbis.a
     # libogg.a
   	 libboost_system.a
   	 libboost_filesystem.a
   	 libboost_serialization.a
   	 libboost_math_tr1.a
     libboost_iostreams.a
     libtbb.so # shared library is recommended officially
     libtbbmalloc.so
     libtbbmalloc_proxy.so
     libippi.a
     libipps.a
     libippcore.a
     libippvm.a
     libpng.a
     libz.a
     # liblapack.so # require for armadillo
     # libblas.so # required for armadillo
     # libarmadillo.so # only shared library available in apt
     pthread
     m
     dl
  )
  link_directories(
    /opt/intel/ipp/lib/intel64
  )
  include_directories(
  	${BAKUAGE_PREBUILT_DIR}/include
    /opt/intel/ipp/include
  )
  set(TINY_PROCESS_SOURCES
    deps/tiny-process-library/process.hpp
    deps/tiny-process-library/process.cpp
    deps/tiny-process-library/process_unix.cpp
  )
endif()

add_subdirectory(deps/googletest)
set(GFLAGS_BUILD_gflags_LIB ON)
add_subdirectory(deps/gflags)

if(WIN32)
else()
  # optim
  # https://stackoverflow.com/questions/5971921/building-a-library-using-autotools-from-cmake
  ExternalProject_Add(optim
      PREFIX ${CMAKE_CURRENT_SOURCE_DIR}/deps/optim
      SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/optim
      BINARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/deps/optim
      CONFIGURE_COMMAND true
      BUILD_COMMAND ${CMAKE_CURRENT_SOURCE_DIR}/deps/optim/configure --header-only-version
      INSTALL_COMMAND true)
endif()

message("BAKUAGE_LINK_LIBRARIES: ${BAKUAGE_LINK_LIBRARIES}")

include_directories(
  src
  deps/bakuage/include
  deps/eigen
  deps/tiny-process-library
  deps/CImg
  deps/libsimdpp
  deps/hnsw
  deps/googletest/googletest/include
  deps/optim/header_only_version
  deps/picojson
)

file(GLOB BAKUAGE_SOURCES deps/bakuage/src/*)
message("BAKUAGE_SOURCES: ${BAKUAGE_SOURCES}")
add_library(bakuage STATIC ${BAKUAGE_SOURCES})
target_link_libraries(bakuage
  ${BAKUAGE_LINK_LIBRARIES}
)
if(WIN32)
else()
add_dependencies(bakuage optim)
endif()

add_executable(build_test src/build_test/main.cpp)
target_link_libraries(build_test
  bakuage
)

file(GLOB BAKUAGE_SND_FILE_INFO_SOURCES src/snd_file_info/*)
message("BAKUAGE_SND_FILE_INFO_SOURCES: ${BAKUAGE_SND_FILE_INFO_SOURCES}")
add_executable(snd_file_info ${BAKUAGE_SND_FILE_INFO_SOURCES})
target_link_libraries(snd_file_info
  bakuage
  gflags::gflags
)

file(GLOB BAKUAGE_AUDIO_ANALYZER_SOURCES src/audio_analyzer/*)
message("BAKUAGE_AUDIO_ANALYZER_SOURCES: ${BAKUAGE_AUDIO_ANALYZER_SOURCES}")
add_executable(audio_analyzer ${BAKUAGE_AUDIO_ANALYZER_SOURCES} ${TINY_PROCESS_SOURCES})
target_link_libraries(audio_analyzer
  bakuage
  gflags::gflags
)

file(GLOB BAKUAGE_TEST_SOURCES src/test/*)
message("BAKUAGE_TEST_SOURCES: ${BAKUAGE_TEST_SOURCES}")
add_executable(test ${BAKUAGE_TEST_SOURCES})
target_link_libraries(test
  bakuage
  gtest
  gflags::gflags
)

file(GLOB BAKUAGE_PHASE_LIMITER_SOURCES src/phase_limiter/*)
message("BAKUAGE_PHASE_LIMITER_SOURCES: ${BAKUAGE_PHASE_LIMITER_SOURCES}")
add_executable(phase_limiter ${BAKUAGE_PHASE_LIMITER_SOURCES} ${TINY_PROCESS_SOURCES})
target_link_libraries(phase_limiter
  bakuage
  gflags::gflags
)

file(GLOB BAKUAGE_SPECTROGRAM_SOURCES src/spectrogram/*)
message("BAKUAGE_SPECTROGRAM_SOURCES: ${BAKUAGE_SPECTROGRAM_SOURCES}")
add_executable(spectrogram ${BAKUAGE_SPECTROGRAM_SOURCES})
target_link_libraries(spectrogram
bakuage
gflags::gflags
)

file(GLOB BAKUAGE_AUDIO_VISUALIZER_SOURCES src/audio_visualizer/*)
message("BAKUAGE_AUDIO_VISUALIZER_SOURCES: ${BAKUAGE_AUDIO_VISUALIZER_SOURCES}")
add_executable(audio_visualizer ${BAKUAGE_AUDIO_VISUALIZER_SOURCES})
target_link_libraries(audio_visualizer
bakuage
gflags::gflags
)

file(GLOB BAKUAGE_CLEAR_MIXER_SOURCES src/clear_mixer/*)
message("BAKUAGE_CLEAR_MIXER_SOURCES: ${BAKUAGE_CLEAR_MIXER_SOURCES}")
add_executable(clear_mixer ${BAKUAGE_CLEAR_MIXER_SOURCES})
target_link_libraries(clear_mixer
bakuage
gflags::gflags
)

file(GLOB BAKUAGE_EFFECT_TEST_SOURCES src/effect_test/*)
message("BAKUAGE_EFFECT_TEST_SOURCES: ${BAKUAGE_EFFECT_TEST_SOURCES}")
add_executable(effect_test ${BAKUAGE_EFFECT_TEST_SOURCES})
target_link_libraries(effect_test
bakuage
gflags::gflags
)

if(DISABLE_BENCH)
else()
file(GLOB BAKUAGE_BENCH_SOURCES src/bench/* src/phase_limiter/Grad*.cpp)
message("BAKUAGE_BENCH_SOURCES: ${BAKUAGE_BENCH_SOURCES}")
add_executable(bench ${BAKUAGE_BENCH_SOURCES})
target_link_libraries(bench
bakuage
libbenchmark.a
)
endif()
