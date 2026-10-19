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

- pipenv (python 3)
- ffmpeg (command line program)
- system audio/image libraries (libsndfile, libpng, zlib and their codec dependencies)

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
It uses Debian 12's GCC 12 / Boost 1.74 toolchain. The SDKs that must retain
older APIs are pinned by URL and SHA-256 in `environment/dependencies.json`:
Linux TBB 2020.3, Armadillo 10.1.2, and IPP 2021.9. Windows uses IPP 2019.2
and TBB 2020.2, plus the conda environment in `environment/windows.yml`.
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
  -DIPPROOT="$PWD/build/dependencies/opt/intel/oneapi/ipp/2021.9.0"
```

The scripts require Python 3.9+; Windows SDK extraction additionally needs
Python 3.11.8+ with tar extraction filters. Linux extraction uses `dpkg-deb`.
An x86-64 toolchain is required. It writes only to the requested prefix and
cache. Normal CMake configure/build does not download or modify SDKs.

On Windows, use a VS 2022 x64 toolchain and an activated environment created
from `environment/windows.yml`. Then fetch the Windows SDKs into
`build/dependencies` with the same script's `--platform windows` option. Pass
both `build/dependencies/windows` and its `Library` subdirectory through
`CMAKE_PREFIX_PATH`, and set `TBB_ROOT` to that `Library` directory. The workflow
in `.github/workflows/build-win.yml` contains the complete configure commands.
The Intel conda channel index is not needed: the verified SDK archives are
fetched directly. Other conda package versions are pinned in the environment
file; transitive dependencies are still resolved by conda.

### Build prerequisites

- CMake 3.23 or newer and a C++11-capable GCC, Clang, or MSVC compiler.
- An x86/x86-64 machine with AVX2/FMA, or AVX with `SANDYBRIDGE_SUPPORT=ON`.
  ARM/Apple Silicon native builds require DSP source changes. On Apple Silicon,
  use an x86-64 toolchain and x86-64 dependencies under Rosetta.
- Intel IPP, legacy Intel TBB **2019/2020** (including tbbmalloc), Boost 1.65+
  (filesystem, serialization, system, math_tr1, iostreams), libsndfile, libpng,
  zlib, and Armadillo headers compatible with the pinned optim submodule.
  oneTBB 2021+ removes APIs used by the current source. Updating CMake does not
  make the DSP code compatible with newer TBB/Armadillo releases.
- Google Benchmark, only when benchmarks are enabled.

The pinned gflags, GoogleTest, and header dependencies are git submodules:

```sh
git submodule update --init --recursive
```

CMake discovers installed libraries rather than assuming fixed library names or
link directories. Use `CMAKE_PREFIX_PATH` for custom installation prefixes,
`IPPROOT` for IPP, and `TBB_ROOT` (or `TBB_DIR` for its CMake package) for legacy
TBB. `CONDA_PREFIX` and `CONDA_ROOT` are also recognized. On Windows, the bundled
64-bit libsndfile import library is a fallback when no installed libsndfile is
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
| `PHASELIMITER_WARNINGS_AS_ERRORS` | `OFF` | Treat project warnings as errors |
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

The Ruby specs accept the binary directory explicitly; they do not need an
install into the source tree. CI uses the separately locked test/lint gems in
`environment/Gemfile` (the historical application Gemfile is unchanged):

```sh
export BUNDLE_GEMFILE="$PWD/environment/Gemfile"
export BUNDLE_PATH="$PWD/build/gems"
bundle install
PHASELIMITER_BIN_DIR="$PWD/build/release/bin" bundle exec rspec
bundle exec rubocop --lint
python3 -B -m unittest discover -s tests/build -v
```

CTest's CLI smoke check reads the bundled WAV using `snd_file_info` and checks
its JSON metadata. It replaces the old empty `build_test` program.

### Install and package

```sh
cmake --install build/release --config Release --prefix /your/install/prefix
```

This installs the seven CLI tools. Windows installation resolves DLL imports,
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
It deliberately excludes any old `resource/sound_quality2_cache` in the source
tree; supply a newly generated one with `--sound-quality-cache build/data/sound_quality2_cache`.
Release stages reject nonempty destinations to avoid silently retaining stale
binaries. Neither staging nor archiving modifies the original resources.
`script/install_mac.sh` still supports the existing `/etc/phaselimiter` layout.

### CI

GitHub Actions is the only CI configuration. `script/ci_linux.sh` builds
Release and Debug, runs CTest, and for Release runs Ruby specs, read-only lint,
packaging tests, CLI integration and staging. Dataset-based validation uses the
commit pinned in `environment/dataset.json`; checkout and generated caches live
under `build/`. Benchmark, historical regression-sweep, and valgrind jobs use
`script/ci_checks.py`. Failures propagate to CI; no benchmark errors are ignored.
Binaries remain uncompressed for diagnostics and tests, and the release archive
provides compression. The old UPX and OOM/cache-dropping workarounds are removed.

Windows CI builds Release with unit tests enabled and runs CTest before CLI
integration and packaging. CTest logs and a JUnit report are uploaded even when
tests fail.

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
