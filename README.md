## phaselimiter

A high quality audio limiter and an automated mastering algorithm written in C++. 

The limiter and automated mastering engine used by Onatsu Bakuage-kun.

## FFT library performance comparison

- Libraries: IPP 2021.9.0 / 2026.1.0, FFTW 3.3.10, PFFFT, PocketFFT, and Rust's RealFFT 3.5.0 (RustFFT 6.4.1).
- Conditions: a Linux VM with an Intel Core i9-9980HK-class CPU, AVX2/FMA, and one thread. Real FFTs of 256–16,384 points (powers of two), single and double precision, forward and inverse transforms. Repeated measurements exclude plan creation but include input copying. C++ uses optimized builds; Rust uses release builds.
- Results: IPP was generally fast, with little difference between the two versions. Alternatives worth considering are PFFFT for small single-precision transforms and FFTW for larger transforms or double precision. RealFFT performed close to FFTW. PocketFFT was slower through the public API used in this comparison.
- Small differences are indicative only because VM performance varies. Windows, macOS, and ARM performance was not measured.

## Global optimization library benchmark

- Solvers: DE-PRMM from the repository's pinned optim and the latest upstream optim, active CMA-ES from libcmaes, jDE from pagmo, MLSL-LDS + Sbplx from NLopt, and global_function_search from dlib.
- Conditions: this repository's mastering5 objective on Linux, with 72 variables and the original regularization and penalties. Twelve audio excerpts (about 6–10 seconds) cover the existing test audio, piano, solo instruments, vocals, orchestra, pop, funk, and electronic music. Each solver used three seeds and a budget of up to 10,000 evaluations, including the baseline. Evaluations within each trial were sequential; the comparison measures objective reduction against evaluation count, not elapsed time.
- Results: using the median across three seeds for each excerpt, CMA-ES ranked first on 11 of 12 excerpts at 10,000 evaluations; jDE won on the remaining glockenspiel excerpt. CMA-ES's worst rank was a tie for third. On 11 excerpts, even its worst seed outperformed the current optim's best seed. At 500–1,000 evaluations, however, NLopt ranked first on all 12 excerpts. Every dlib trial on the additional excerpts reached the 300-second timeout, so dlib was excluded from rankings at equal evaluation budgets.
- These are practical configurations: the current optim retains its original unbounded search, while the other solvers share a bounded, normalized search domain. With only three trials per excerpt, the results do not establish general superiority, convergence to a global optimum, or perceived audio quality.

### Using CMA-ES in mastering5

Select active CMA-ES with:

```sh
phase_limiter --input=input.wav --output=output.wav \
  --mastering --mastering_mode=mastering5 \
  --mastering5_optimization_algorithm=cmaes \
  --mastering5_optimization_seed=1 \
  --mastering5_optimization_max_eval_count=10000
```

This mode uses [libcmaes](https://github.com/CMA-ES/libcmaes) v0.10.2,
built from the pinned submodule with the repository's Eigen headers. Run
`git submodule update --init --recursive` before building an updated checkout.
The default solver remains `de_prmm`.

CMA-ES starts from the unchanged effect parameters and searches within the
mastering5 bounds, normalized to `[0, 1]` per parameter, with an initial step
size of 0.2 and automatic population size. Objective evaluations use libcmaes'
OpenMP parallel loop. `--worker_count` controls the thread limit (zero selects
the same automatic CPU count as the other processing stages), capped by the
population size. Candidate generation and covariance updates remain sequential.
The same seed produces the same search across worker counts within the same
build when objective evaluations are deterministic; callback/log order can vary.
The positive evaluation budget includes the initial baseline and is a strict
upper limit, even for a partial final generation; convergence can stop earlier.
The best evaluated point is retained, with the baseline as a fallback.
Seeds, including zero, are reproducible within the same build. These settings
do not imply bit-identical reproduction of the historical benchmark above.
The build applies a checked header patch to give each CMA-ES sampler its own
random engine and Gaussian distribution state. Independent concurrent or nested
searches therefore do not share random state. This fix changes the random
sequence from the original integration; old seeds do not reproduce old results.
An OpenMP-capable C++ compiler and runtime are required to build this mode.

## Phase limiter parallelization

The limiter schedules boundary FFT work as soon as its overlapping interior
regions finish. See the [parallelization research notes](docs/phase-limiter-parallelization.md)
(in Japanese) for the implementation, benchmark comparisons, source FFT cache
tradeoffs, and reproduction instructions.

For `--mastering --mastering_mode=mastering5`, the default `de_prmm` solver
accepts `--mastering5_optimization_seed=1` (also the default). Each population
slot has its own random stream, so changing `--worker_count` does not change
candidate generation for the same seed. The seed option also applies to `cmaes`.
The old shared random sequence is not preserved. Reproducibility is tested
within the same build; it does not promise identical floating-point results
across platforms, callback/log ordering, or bit-identical final WAV files.


## project status

inactive

- you can create issues and PRs.
- I can respond about once every three months.

## license

MIT

## Installation

todo

prebuilt linux and windows executables are available here

https://github.com/ai-mastering/phaselimiter/releases

## runtime dependencies

- ffmpeg (command line program)
- system audio/image libraries (libsndfile, libpng, zlib and their codec dependencies)

Native tools require neither Python nor Pipenv. Development and release tooling
uses Python 3.

`phase_limiter --disable_input_encode` requires a stereo WAV at 44.1 kHz;
other channel counts or sample rates are rejected before processing. With input
conversion enabled (the default), FFmpeg converts the input to that format.
`--ffmpeg` accepts an executable path or a name on PATH, without shell quoting.
Final audio and `--output_after_pre_compression` are staged beside their
respective destinations before replacement, so a failed copy or replacement
preserves existing output files.

## For developers

### Reproducible build environment

The Dockerfile's `build` target is shared by local development and Linux CI.
It uses Debian 12's GCC 12 / Boost 1.74 toolchain. The SDKs are pinned by URL
and SHA-256 in `environment/dependencies.json`:
Linux oneTBB 2023.1, Armadillo 10.1.2, and IPP 2021.9. Windows uses IPP 2019.2
and oneTBB 2023.1, plus the conda environment in `environment/windows.yml`.
OS package security updates are allowed; this is not a bit-for-bit OS snapshot.

For an x86-64 Linux build (also usable with Docker's x86-64 emulation):

```sh
docker build --platform linux/amd64 --target build -t phaselimiter-build .
docker run --rm --platform linux/amd64 \
  --user "$(id -u):$(id -g)" \
  --mount "type=bind,src=$PWD,dst=/workspace" --workdir /workspace \
  -e HOME=/tmp phaselimiter-build bash -euc '
    cmake --preset release -G Ninja
    cmake --build --preset release --parallel 2
    ctest --preset release
  '
```

The development target adds interactive tools and Codex. The devcontainer uses
that target; `bash script/linux_shell.sh` also builds and opens it. These
containers contain the SDKs already; no manual setup inside a running container
is required. The bootstrap verifies archive checksums before extraction.

For a native Linux setup with matching system prerequisites installed:

```sh
python3 script/fetch_build_dependencies.py --platform linux \
  --prefix build/dependencies --cache-dir build/downloads
cmake --preset release \
  -DCMAKE_PREFIX_PATH="$PWD/build/dependencies/legacy/usr" \
  -DTBB_ROOT="$PWD/build/dependencies/opt/intel/oneapi/tbb/2023.1" \
  -DIPPROOT="$PWD/build/dependencies/opt/intel/oneapi/ipp/2021.9.0"
```

When upgrading from legacy TBB, use a fresh build directory to avoid stale
headers or library paths in the CMake cache.

The scripts require Python 3.9+; Windows SDK extraction additionally needs
Python 3.11.8+ with tar extraction filters. Linux extraction uses `dpkg-deb`.
An x86-64 toolchain is required. It writes only to the requested prefix and
cache. Normal CMake configure/build does not download or modify SDKs.

Windows CLI runtime support requires Windows 10 version 1903 or later, or
Windows 11. Each CLI embeds an `activeCodePage=UTF-8` process manifest;
users do not need to enable the system-wide UTF-8 locale option. Internal narrow
path strings are UTF-8, including executable, input, output, cache, and temporary
paths. Convert explicitly at library boundaries that require another encoding.
Japanese path support via this manifest is not guaranteed on older Windows.
See [Microsoft's UTF-8 code page documentation](https://learn.microsoft.com/en-us/windows/apps/design/globalizing/use-utf8-code-page).

`script/test_windows_utf8.py --stage-dir build/stage/windows/phaselimiter
--ffmpeg build/tools/ffmpeg/ffmpeg.exe` checks the embedded manifests and runs
WAV/MP3/AAC mastering with Japanese, spaces, and emoji in distribution, FFmpeg,
input, output, cache, and temporary paths. Run on native Windows with the
system-wide UTF-8 option disabled, including Windows 10 1903 for minimum-version
acceptance. Wine alone does not establish native Windows compatibility.

On Windows, use a VS 2022 x64 toolchain and an activated environment created
from `environment/windows.yml`. Then fetch the Windows SDKs into
`build/dependencies` with the same script's `--platform windows` option. Pass
both `build/dependencies/windows` and its `Library` subdirectory through
`CMAKE_PREFIX_PATH`, and set `TBB_ROOT` to
`build/dependencies/windows/oneapi-tbb-2023.1.0`. The workflow
in `.github/workflows/build-win.yml` contains the complete configure commands.
The Intel conda channel index is not needed: the verified SDK archives are
fetched directly. Other conda package versions are pinned in the environment
file; transitive dependencies are still resolved by conda.

### Native macOS ARM64

Install Xcode Command Line Tools (`xcode-select --install`) and an ARM64
Miniforge installation, then create and activate the dependency environment:

```sh
conda env create -f environment/macos.yml
conda activate phaselimiter-build
git submodule update --init --recursive
cmake -S . -B build/macos -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=/usr/bin/clang -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
  -DCMAKE_OSX_ARCHITECTURES=arm64 -DCMAKE_OSX_DEPLOYMENT_TARGET=11.0 \
  -DBUILD_TESTING=ON -DPHASELIMITER_STATIC_BOOST=OFF \
  "-DCMAKE_PREFIX_PATH=$CONDA_PREFIX" \
  "-DOpenMP_CXX_FLAGS=-Xpreprocessor -fopenmp" \
  "-DOpenMP_CXX_INCLUDE_DIR=$CONDA_PREFIX/include" \
  -DOpenMP_CXX_LIB_NAMES=omp \
  "-DOpenMP_omp_LIBRARY=$CONDA_PREFIX/lib/libomp.dylib"
cmake --build build/macos --parallel "$(sysctl -n hw.logicalcpu)"
ctest --test-dir build/macos --output-on-failure
```

CLI integration tests require `ffmpeg` on `PATH`, including its `libmp3lame`
encoder. `imageio_ffmpeg.get_ffmpeg_exe()` locates the executable supplied by
this environment's Python wheel; add its directory to `PATH` if needed.
Keep the conda environment active when running the binaries.

The public `bakuage::` interfaces select IPP on x86 and Accelerate on macOS
ARM64. Complex FFTs use vDSP for power-of-two lengths and Eigen's bundled
KissFFT for other lengths. Forward and inverse transforms remain unnormalized;
real FFT CCS/Pack/Perm layouts and the orthonormal 2D DCT match the IPP API.
NEON retains eight-lane logical vectors in the limiter and preserves its NaN
clipping rule. No fast-math options are enabled. Floating-point reduction and
library implementation differences can still change optimization trajectories
across architectures, so bit-identical final audio is not guaranteed.

The `DspBackend` unit tests check transforms, packed layouts, in-place paths,
complex arithmetic, half conversion, and FIR history against independent
references. The macOS CI job also loads a model cache generated by Linux CI
and exercises mastering. Linux Release also records a per-commit DSP reference
with the original-input analyses, final limiter/mastering5 audio, convergence
diagnostics, and timings. Both native macOS jobs automatically compare against
that artifact using the same cache, inputs, one worker, and deterministic
Nelder-Mead mastering optimization. This does not require a local Mac or QEMU.

`script/check_dsp_port.py` checks original-input levels within 0.1 dB and the
model score within 0.005. Final-output levels allow 0.25 dB and a score difference
of 0.02; the limiter waveform allows 2% relative RMS error. The final normalized
limiter error may be at most 10% worse for Linux x86-64 -> macOS ARM64, and 5%
for other comparisons, including Intel Mac. Both retain a 1e-4 absolute allowance
near zero. Mastering waveform differences and both iteration counts are reported
as well.
These bounds allow floating-point/optimization differences while rejecting large
numerical regressions; bit equality across IPP and Accelerate is not required.
Each command is warmed up once and measured three times. The median timings,
audio SNR, and comparison result appear in the macOS Actions job summary, with
full JSON, output WAVs, and logs in its results artifact. The Linux reference is
also available as `linux-dsp-reference`. Different runner hardware makes these
timings informational rather than an automatic speed regression threshold.
Within one validation job, the warmup and every measured execution must produce
identical analysis JSON values, decoded float PCM samples, and reported convergence
diagnostics. Differences fail CI independently of the cross-platform tolerances.
The report includes per-execution fingerprints, binary/input/cache hashes, CPU
model, and normalization diagnostics. This tests repeated execution with a fixed
binary, cache, CPU, and libraries; separate GitHub runner machines can have
different CPUs and math-library dispatch paths, so cross-job bit equality is not
promised.

Model preparation sorts analysis paths before PCA and HNSW insertion. The
`sound-quality-cache-repeatability` CTest checks that filesystem traversal order
and worker count do not change the generated cache bytes on the same platform.
Cross-compilation alone does not validate native execution or performance.

### Native Intel Mac

Use an x86-64 Miniforge installation and the same `environment/macos.yml`
dependency environment as above. Intel Mac retains the IPP backend and AVX2/FMA;
Apple Silicon continues to use Accelerate and NEON. Intel Mac IPP is pinned to
2021.9.0, matching the Linux version, and downloaded from Intel's Anaconda
channel with SHA-256 verification:

```sh
python script/fetch_build_dependencies.py --platform macos-intel \
  --prefix build/dependencies --cache-dir build/downloads
```

Use the macOS configure command above with these two arguments:

```sh
-DCMAKE_OSX_ARCHITECTURES=x86_64 \
"-DIPPROOT=$PWD/build/dependencies/macos-intel"
```

Replace the ARM64 architecture argument with `x86_64` and add `IPPROOT`.
All other build and test commands are shared. Set `SANDYBRIDGE_SUPPORT=ON`
for an older Intel Mac that supports AVX but lacks AVX2/FMA.

The macOS CI matrix runs natively on `macos-15` (ARM64/Accelerate) and
`macos-15-intel` (x86-64/IPP). Both execute CTest and CLI integration tests
using a model cache generated by Linux CI. The Intel job additionally
covers the AVX2/FMA configuration; Linux and Windows CI retain their AVX builds.

### Build prerequisites

- CMake 3.23 or newer and a C++11-capable GCC, Clang, or MSVC compiler.
- An x86/x86-64 machine with AVX2/FMA, or AVX with `SANDYBRIDGE_SUPPORT=ON`.
  Native macOS ARM64 builds use Accelerate and NEON; IPP is not required.
- Intel IPP on x86, oneTBB **2021.8+** (including tbbmalloc), Boost 1.65+
  (filesystem, serialization, system, math_tr1, iostreams), libsndfile, libpng,
  zlib, and Armadillo headers compatible with the pinned optim submodule.
  The SDK manifest pins oneTBB 2023.1. The optim fork retains its own parallel
  loops; its oneTBB-compatible headers are generated in the build directory.
- Google Benchmark, only when benchmarks are enabled.

The pinned gflags, GoogleTest, and header dependencies are git submodules:

```sh
git submodule update --init --recursive
```

HNSW is pinned to upstream hnswlib v0.9.0. Existing model caches retain the
legacy 64-bit little-endian index layout used on x86-64 and ARM64; upgrading
does not require rebuilding them. A frozen legacy index is covered by the
`hnsw-compatibility` test.

CMake discovers installed libraries rather than assuming fixed library names or
link directories. Use `CMAKE_PREFIX_PATH` for custom installation prefixes,
`IPPROOT` for IPP, and `TBB_ROOT` (or `TBB_DIR` for its CMake package) for
oneTBB. `CONDA_PREFIX` and `CONDA_ROOT` are also recognized. On Windows, the
bundled 64-bit libsndfile import library is a fallback when no installed libsndfile is
found. Keep all dependencies on the same architecture and MSVC runtime ABI.

### Configure, build, and test

Run from the repository root:

```sh
cmake --preset release
cmake --build --preset release --parallel 2
ctest --preset release
```

`debug` and `release-avx` presets are also provided. Presets use the platform's
default generator; add `-G Ninja` or `-G Xcode` on the first configure to choose
another one. Use a fresh build directory when changing generators or compilers.
Build products live under `build/<preset>/bin` and `lib`; multi-configuration
builds (Visual Studio/Xcode) add `Release` or `Debug` below those directories.
Start with low parallelism: some audio translation units use substantial memory.

Without presets, the equivalent commands are:

```sh
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release
cmake --build build/local --config Release --parallel 2
ctest --test-dir build/local -C Release --output-on-failure
```

For custom dependency locations, for example:

```sh
cmake --preset release -DIPPROOT=/path/to/ipp \
  -DCMAKE_PREFIX_PATH="/path/to/legacy-tbb;/path/to/other/dependencies"
```

Do not configure in the source directory. If an old checkout contains a root
`CMakeCache.txt`, `CMakeFiles`, or generated Makefile/project, remove those
old generated files first. `optim` headers are now generated in the build tree
on all platforms; its submodule and the old Windows prebuilt headers are not
modified or used as a generation destination.

| Option | Default | Purpose |
| --- | --- | --- |
| `BUILD_TESTING` | `ON` | Build the C++ tests and run the CLI smoke check |
| `PHASELIMITER_BUILD_BENCHMARKS` | `OFF` | Build `bench`; requires Google Benchmark |
| `SANDYBRIDGE_SUPPORT` | `OFF` | Target AVX rather than AVX2/FMA |
| `PHASELIMITER_WARNINGS_AS_ERRORS` | `ON` | Treat project warnings as errors (GCC/Clang only; MSVC keeps warnings nonfatal) |
| `PHASELIMITER_STATIC_BOOST` | Unix: `ON`, Windows: `OFF` | Prefer static Boost libraries |
| `PHASELIMITER_TEST_DATA_DIR` | empty | Enable external dataset validation |

The old `DISABLE_TARGET_TEST` and `DISABLE_TARGET_BENCH` flags are accepted for
existing command lines, but now affect tests and benchmarks independently.
Use the options above for new configurations.

The test executable is now named `phaselimiter_tests`: `test` conflicts with
CTest's build target under Ninja. CTest runs it from the repository root so
relative resource paths work. It excludes the known crashing `JsFilterTest*`
suite (as the existing CI did), and separates dataset validation from unit tests.
To include validation against an existing bakuage_dataset1 checkout:

```sh
cmake --preset release \
  -DPHASELIMITER_TEST_DATA_DIR=/path/to/bakuage_dataset1/analysis_shortpath
ctest --preset release
```

### Datasets

- [`bakuage_dataset1/analysis_shortpath`](https://github.com/ai-mastering/bakuage_dataset1)
  is the larger analysis dataset used for sound_quality2 cache generation and
  dataset validation. The cache is used by `audio_analyzer` and `mastering5`.
  CI fetches the commit pinned in `environment/dataset.json`.
- `resource/analysis_data` is the smaller bundled dataset. Its
  `resource/detection_preparation*.json` files define the general/pop/jazz/classical
  reference sets and pairwise distances for legacy `detect2` reference selection.
  The bundled JSONs also support small sound_quality2 test caches; they are
  separate from the larger external dataset.

The CLI regression tests use Python's standard-library `unittest`. They cover
limiter smoke cases, decoded frame counts across AAC/MP3/WAV conversions, and
audio_analyzer reference values. CTest runs them in Release builds, matching
the former RSpec suite. They can also be run directly against any build:

```sh
python3 -B script/test_cli_regression.py --bin-dir build/release/bin -v
python3 -B -m unittest discover -s tests/build -v
```

CTest's CLI smoke check reads the bundled WAV using `snd_file_info` and checks
its JSON metadata.

### Install and package

```sh
cmake --install build/release --config Release --prefix /your/install/prefix
```

This installs the eight CLI tools. Windows installation resolves DLL imports,
including transitive DLL dependencies, using the active conda environment,
configured SDK prefixes, and `PHASELIMITER_RUNTIME_DIRS` when extra search paths
are needed. Windows system DLLs are excluded. The target machine needs the
Visual C++ runtime. Linux installation bundles non-system shared dependencies
(such as legacy TBB) into `lib`; executables use an install RPATH to find them.
System media libraries and ffmpeg remain runtime prerequisites.

Stage a release in a **new or empty** directory, without writing binaries or
caches into the source tree:

```sh
python3 script/stage_release.py --build-dir build/release \
  --destination build/stage/linux --config Release
python3 script/package_release.py --stage-dir build/stage/linux \
  --output build/artifacts/release.tar.xz
```

Use a `.zip` output for Windows. Both archive formats retain a `phaselimiter/`
root. Staging uses `packaging/release_files.txt` and `packaging/licenses`.
Archives contain native tools, runtime libraries, resources, README, and licenses.
Development scripts and test audio remain in the source repository and are not
distributed. Legacy Python/Pipenv manifests have been removed. No Python or
Pipenv setup is needed on the target machine.
It deliberately excludes any old `resource/sound_quality2_cache` in the source
tree; supply a newly generated one with `--sound-quality-cache build/data/sound_quality2_cache`.
Release stages reject nonempty destinations to avoid silently retaining stale
binaries. Neither staging nor archiving modifies the original resources.
Extract the archive and run tools directly from `phaselimiter/bin/`.
Keep the extracted directory layout intact so executables can find the bundled
libraries in `phaselimiter/lib/`.

### CI

GitHub Actions is the only CI configuration. `script/ci_linux.sh` builds
Release and Debug, runs CTest (including CLI regression tests in Release), and
for Release runs packaging tests, CLI integration and staging. Dataset-based validation uses the
commit pinned in `environment/dataset.json`; checkout and generated caches live
under `build/`. Benchmark, historical regression-sweep, and valgrind jobs use
`script/ci_checks.py`. Failures propagate to CI; no benchmark errors are ignored.
Binaries remain uncompressed for diagnostics and tests, and the release archive
provides compression. The old UPX and OOM/cache-dropping workarounds are removed.

Windows CI builds Release with unit tests enabled and runs CTest before CLI
integration and packaging. CTest logs and a JUnit report are uploaded even when
tests fail.

Linux Release, Debug, and sanitizer builds cache compilation results separately
with ccache and log per-build hit statistics. Windows caches legacy SDK archives;
the fetch script still verifies their pinned checksums before extraction.

The Linux sanitizer job builds project code (including `deps/bakuage`) with
ASan, UBSan, and libstdc++ assertions, then runs CTest. GCC builds additionally
enable `-fsanitize=bounds-strict`; Clang builds use the shared ASan/UBSan settings.
Reproduce it with:

```sh
cmake -S . -B build/sanitizers -G Ninja -DCMAKE_BUILD_TYPE=Debug \
  -DSANDYBRIDGE_SUPPORT=ON -DBUILD_TESTING=ON -DPHASELIMITER_SANITIZERS=ON \
  -DCMAKE_EXE_LINKER_FLAGS=-no-pie
cmake --build build/sanitizers --parallel 2
ASAN_OPTIONS=detect_leaks=1:halt_on_error=1 \
UBSAN_OPTIONS=halt_on_error=1:print_stacktrace=1 \
  ctest --test-dir build/sanitizers --output-on-failure
```

Sanitizers require GCC or Clang. The Linux CI example links non-PIE sanitizer
executables to avoid ASan startup address-layout conflicts on newer hosts;
normal release builds retain their default linker settings. Prebuilt DSP SDK
internals are not instrumented.
CTest includes WAV metadata checks for empty/short mono and stereo files at
44.1/48 kHz, plus DSP tests for empty blocks, SIMD/block tails, scalar FIR
convolution, delay behavior, short interleaved peak/RMS input, and minimum
resampler blocks. These are module
contracts; they do not establish that every CLI accepts empty audio.
Valgrind uses separate Debug binaries with full leak checking and origin
tracking. Definite and indirect leaks fail the job. Possible leaks and
still-reachable allocations are reported for review without failing it, because
legacy TBB workers can leave thread-local storage classified as possibly lost
at process exit. Logs are uploaded even on failure.
The existing `JsFilterTest*` and dataset-validation exclusions in the default
unit suite remain; the Valgrind suite additionally excludes `ClearMixerFilter*`.

All CI-generated binaries, SDK downloads, datasets, caches, stages and reports
stay under `build/`. CI also checks that source files and submodules remain
clean. No CI step runs automatic source formatting. Docker and Windows builds
must be exercised on their respective CI runners; Linux-native checks alone do
not validate those complete environments.

CMake references: [presets](https://cmake.org/cmake/help/v3.23/manual/cmake-presets.7.html)
and [imported targets](https://cmake.org/cmake/help/v3.23/guide/importing-exporting/index.html).


## Audio detector

`audio_detector` implements the legacy Python `detect2` task in C++. It accepts
an `audio_analyzer` analysis JSON and selects a classic mastering reference:

```sh
build/release/bin/audio_detector detect2 \
  --source input_analysis.json \
  --detection_preparation resource/detection_preparation.json \
  --neighbor_ratio 0.01 --outlier_ratio 0.1
```

`--task=detect2` is also accepted. Stdout contains the original result JSON:
`lof`, `border_lof`, `sound_quality`, and the one-entry `ranking` with `path` and
`lof`. Errors go to stderr with a nonzero exit status. The returned path is the
unchanged preparation entry. General/pop/jazz/classical select different
`resource/detection_preparation*.json` files; there is no genre classification.

Defaults match the Python detector: neighbor ratio `0.03` and outlier ratio `0.1`.
LOF distances absent from the cache use a fixed covariance regularization of
`0.001`; final ranking uses the value stored in the preparation. The former Rails server explicitly used
neighbor ratio `0.01`, as in the example. The preparation defaults to
`./resource/detection_preparation.json`. Relative candidate paths are resolved
beside the preparation, including bundled legacy `./resource/analysis_data/...`
paths; the old working-directory-relative location is a fallback. Absolute paths
are used verbatim. This also supports anonymous `./analysis_data/000001.json`
entries without rewriting the preparation or depending on the working directory.

The independent `src/classic_detector` module uses `bands_short` and
`covariance_short`, subtracts overall loudness from Mid/Side means, and reproduces
scikit-learn 0.19.1 / SciPy 1.0.1 / NumPy 1.14.3 behavior. This includes the
symmetric Gaussian KL distance, brute-force LOF, neighbor tie ordering, the
rounded-first-mean distance-cache shortcut, percentile interpolation, and the
old score formula. NumPy's specialized selection/sort code is included under its
BSD license. Degenerate scores retain Python's `NaN`/`Infinity` JSON extensions.
It does not use project DSP, HNSW, sound_quality2, or a Python runtime. It rebuilds
the LOF state on each invocation rather than reading or writing Python pickle
caches. Audio analysis, phase_limiter invocation, and output-level decisions stay
with the caller. Only `detect2` is supported; existing preparation files are
shipped rather than regenerated by this executable.


### Detector origin and historical tooling

The C++ detector ports `detect2` from the former `script/create_reference.py`.
The original Python implementation, shell launcher, pinned Python dependencies,
preparation generator, and dataset anonymization helpers are preserved in Git at
commit [`078175aaf169e74ed0b1407ad392cbed8b58cd03`](https://github.com/ai-mastering/phaselimiter/tree/078175aaf169e74ed0b1407ad392cbed8b58cd03).
They were removed from the current tree after the native replacement was verified.
To inspect the original implementation locally:

```sh
git show 078175aaf169e74ed0b1407ad392cbed8b58cd03:script/create_reference.py
```

The lockfile in that revision records NumPy 1.14.3, SciPy 1.0.1, and
scikit-learn 0.19.1, which define the detector compatibility target. Legacy Ruby
experiment scripts and their root Gemfile are also available in that revision.
The remaining RSpec tests have since been migrated to Python/CTest.
