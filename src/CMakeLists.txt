# Common CLI requirements. Headers under src are shared by several tools.
add_library(phaselimiter_cli INTERFACE)
target_include_directories(phaselimiter_cli INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")
target_link_libraries(phaselimiter_cli INTERFACE bakuage gflags::gflags)

add_executable(snd_file_info
  snd_file_info/main.cpp
)
target_link_libraries(snd_file_info PRIVATE phaselimiter_cli)
phaselimiter_enable_warnings(snd_file_info)

add_executable(audio_analyzer
  audio_analyzer/main.cpp
  audio_analyzer/sound_quality2_find_nn.cpp
  audio_analyzer/sound_quality2_preparation.cpp
  audio_analyzer/spectrum_distribution_base_image.cpp
  audio_analyzer/test_dft.cpp
  audio_analyzer/test_sound_quality.cpp
)
target_link_libraries(audio_analyzer PRIVATE phaselimiter_cli)
phaselimiter_enable_warnings(audio_analyzer)

# Legacy detect2 compatibility; deliberately independent of bakuage DSP,
# HNSW, IPP, and sound_quality2.
add_library(classic_detector STATIC
  classic_detector/data.cpp
  classic_detector/detection.cpp
  classic_detector/numpy_neighbors.cpp)
target_include_directories(classic_detector PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")
target_include_directories(classic_detector SYSTEM PUBLIC "${PROJECT_SOURCE_DIR}/deps/eigen")
target_include_directories(classic_detector PRIVATE "${PROJECT_SOURCE_DIR}/deps/picojson")
target_link_libraries(classic_detector PUBLIC phaselimiter_options PRIVATE Boost::filesystem)
phaselimiter_enable_warnings(classic_detector)

add_executable(audio_detector audio_detector/main.cpp)
target_link_libraries(audio_detector PRIVATE classic_detector gflags::gflags)
phaselimiter_enable_warnings(audio_detector)
install(FILES "${PROJECT_SOURCE_DIR}/packaging/licenses/numpy.txt"
  DESTINATION "${CMAKE_INSTALL_DATADIR}/phaselimiter/licenses")

add_executable(phase_limiter
  phase_limiter/auto_mastering.cpp
  phase_limiter/auto_mastering2.cpp
  phase_limiter/auto_mastering3.cpp
  phase_limiter/auto_mastering5.cpp
  phase_limiter/enhancement.cpp
  phase_limiter/equalization.cpp
  phase_limiter/freq_expander.cpp
  phase_limiter/main.cpp
  phase_limiter/pre_compression.cpp
  phase_limiter/resampling.cpp
  phase_limiter/test_grad.cpp
  phase_limiter/test_grad_calculator.cpp
  phase_limiter/test_perfect_hash_power_of_2.cpp
)
target_link_libraries(phase_limiter PRIVATE phaselimiter_cli phaselimiter_optim)
phaselimiter_enable_warnings(phase_limiter)

add_executable(spectrogram
  spectrogram/main.cpp
)
target_link_libraries(spectrogram PRIVATE phaselimiter_cli)
phaselimiter_enable_warnings(spectrogram)

add_executable(audio_visualizer
  audio_visualizer/main.cpp
)
target_link_libraries(audio_visualizer PRIVATE phaselimiter_cli)
phaselimiter_enable_warnings(audio_visualizer)

add_executable(clear_mixer
  clear_mixer/main.cpp
)
target_link_libraries(clear_mixer PRIVATE phaselimiter_cli)
phaselimiter_enable_warnings(clear_mixer)

add_executable(effect_test
  effect_test/main.cpp
)
target_link_libraries(effect_test PRIVATE phaselimiter_cli)
phaselimiter_enable_warnings(effect_test)

set(phaselimiter_tools snd_file_info audio_analyzer audio_detector phase_limiter spectrogram
  audio_visualizer clear_mixer effect_test)
if(UNIX AND NOT APPLE)
  set_target_properties(${phaselimiter_tools} PROPERTIES INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()
install(TARGETS ${phaselimiter_tools}
  RUNTIME_DEPENDENCY_SET phaselimiter_runtime
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
include(InstallRuntimeDependencies)

if(BUILD_TESTING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  add_test(NAME auxiliary-cli COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/script/test_auxiliary_cli.py"
    "--bin-dir=$<TARGET_FILE_DIR:audio_visualizer>"
    "--output-dir=${PROJECT_BINARY_DIR}/Testing/auxiliary-cli/$<CONFIG>")
  add_test(NAME audio-file-boundaries COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/script/test_audio_boundaries.py"
    "--tool=$<TARGET_FILE:snd_file_info>")

  add_test(NAME limiter-io COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/script/test_limiter_io.py"
    "--tool=$<TARGET_FILE:phase_limiter>")

  # The former RSpec suite ran in Release CI only.
  if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_CONFIGURATION_TYPES)
    set(cli_regression_configurations)
    if(CMAKE_CONFIGURATION_TYPES)
      set(cli_regression_configurations CONFIGURATIONS Release)
    endif()
    add_test(NAME cli-regression COMMAND "${Python3_EXECUTABLE}"
      "${PROJECT_SOURCE_DIR}/script/test_cli_regression.py"
      "--bin-dir=$<TARGET_FILE_DIR:phase_limiter>" -v
      ${cli_regression_configurations})
    set_tests_properties(cli-regression PROPERTIES TIMEOUT 900)
  endif()

  add_test(NAME cli-smoke COMMAND "${CMAKE_COMMAND}"
    "-DTOOL=$<TARGET_FILE:snd_file_info>"
    "-DINPUT=${PROJECT_SOURCE_DIR}/test_data/test2.wav"
    -P "${PROJECT_SOURCE_DIR}/cmake/TestSndFileInfo.cmake")

  add_executable(optim_rng_tests test/optim_rng.cpp)
  target_link_libraries(optim_rng_tests PRIVATE phaselimiter_cli phaselimiter_optim gtest_main)
  phaselimiter_enable_warnings(optim_rng_tests)
  add_test(NAME optim-rng COMMAND optim_rng_tests)

  add_executable(classic_detector_tests test/classic_detector.cpp)
  target_link_libraries(classic_detector_tests PRIVATE classic_detector gtest_main Boost::filesystem)
  phaselimiter_enable_warnings(classic_detector_tests)
  add_test(NAME classic-detector-unit COMMAND classic_detector_tests)
  add_test(NAME audio-detector-cli COMMAND "${Python3_EXECUTABLE}"
    "${PROJECT_SOURCE_DIR}/script/test_audio_detector.py"
    "--tool=$<TARGET_FILE:audio_detector>")

  # CTest reserves the target name "test" (including Ninja executable aliases).
  add_executable(phaselimiter_tests
    phase_limiter/resampling.cpp
    test/resampling.cpp
    test/aligned_memory.cpp
    test/cli_io.cpp
    test/audio_boundaries.cpp
    test/clear_mixer_filter3.cpp
    test/clear_mixer_filter4.cpp
    test/delay_filter2.cpp
    test/example.cpp
    test/fir_filter_bank.cpp
    test/fir_resample_filter.cpp
    test/grad_tasks.cpp
    test/lof.cpp
    test/loudness.cpp
    test/main.cpp
    test/old_tests.cpp
    test/peak.cpp
    test/reverb.cpp
    test/sound_quality2.cpp
    test/temporary_memory.cpp
    test/utils.cpp
    test/vector_math.cpp
  )
  # Link only headers and GoogleTest: this test supplies a failing allocator.
  add_executable(aligned_vector_failure_tests test/aligned_vector_failure.cpp)
  target_include_directories(aligned_vector_failure_tests PRIVATE "${PROJECT_SOURCE_DIR}/deps/bakuage/include")
  target_link_libraries(aligned_vector_failure_tests PRIVATE gtest_main Boost::boost phaselimiter_options)
  phaselimiter_enable_warnings(aligned_vector_failure_tests)
  add_test(NAME aligned-vector-failure COMMAND aligned_vector_failure_tests)
  # Supply a failing sf_close without invalidating a descriptor owned by the CRT.
  add_executable(sndfile_close_failure_tests test/sndfile_close_failure.cpp)
  target_include_directories(sndfile_close_failure_tests PRIVATE
    "${PROJECT_SOURCE_DIR}/deps/bakuage/include"
    "$<TARGET_PROPERTY:SndFile::SndFile,INTERFACE_INCLUDE_DIRECTORIES>")
  target_link_libraries(sndfile_close_failure_tests PRIVATE gtest_main phaselimiter_options)
  phaselimiter_enable_warnings(sndfile_close_failure_tests)
  add_test(NAME sndfile-close-failure COMMAND sndfile_close_failure_tests)
  add_executable(process_fixture test/process_fixture.cpp)
  add_dependencies(phaselimiter_tests process_fixture)
  target_compile_definitions(phaselimiter_tests PRIVATE
    PROCESS_FIXTURE_PATH="$<TARGET_FILE:process_fixture>")
  # Let leak checkers inspect the process after all legacy TBB workers exit.
  target_compile_definitions(phaselimiter_tests PRIVATE TBB_PREVIEW_WAITING_FOR_WORKERS=1)
  target_link_libraries(phaselimiter_tests PRIVATE phaselimiter_cli gtest)
  phaselimiter_enable_warnings(phaselimiter_tests)
  # The validation suite needs a separately downloaded dataset. JsFilterTest
  # was already excluded by CI because it crashes; do not hide other failures.
  add_test(NAME unit COMMAND phaselimiter_tests
    --gtest_filter=-JsFilterTest*:SoundQuality2Calculator.Validation)
  set_tests_properties(unit PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  set(PHASELIMITER_TEST_DATA_DIR "" CACHE PATH "bakuage_dataset1 analysis directory")
  if(PHASELIMITER_TEST_DATA_DIR)
    add_test(NAME dataset-validation COMMAND phaselimiter_tests
      --gtest_filter=SoundQuality2Calculator.Validation
      "--sound_quality2_analysis_data=${PHASELIMITER_TEST_DATA_DIR}")
    set_tests_properties(dataset-validation PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}")
  endif()
endif()

if(PHASELIMITER_BUILD_BENCHMARKS)
  find_package(benchmark CONFIG REQUIRED)
  add_executable(bench
    bench/dft2.cpp
    bench/fir_filter_bank.cpp
    bench/grad.cpp
    bench/grad_calculator.cpp
    bench/lof.cpp
    bench/main.cpp
    bench/memory.cpp
    bench/vector_math.cpp
  )
  target_link_libraries(bench PRIVATE phaselimiter_cli benchmark::benchmark)
  phaselimiter_enable_warnings(bench)
endif()
