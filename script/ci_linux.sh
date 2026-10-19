#!/bin/bash
# Run inside the Dockerfile's build target, from the repository root.
set -euo pipefail
configuration=${1:?Usage: ci_linux.sh Release|Debug}
case "$configuration" in
  Release) benchmarks=ON ;;
  Debug) benchmarks=OFF ;;
  *) echo "Unsupported configuration: $configuration" >&2; exit 1 ;;
esac
build_dir="build/$configuration"
dataset_option=()
if [ "$configuration" = Release ]; then
  python3 script/prepare_dataset.py --destination build/dataset
  dataset_option=("-DPHASELIMITER_TEST_DATA_DIR=$PWD/build/dataset/analysis_shortpath")
fi
cmake -S . -B "$build_dir" -G Ninja \
  -DCMAKE_BUILD_TYPE="$configuration" -DCMAKE_CXX_COMPILER_LAUNCHER=ccache \
  -DSANDYBRIDGE_SUPPORT=ON -DBUILD_TESTING=ON \
  -DPHASELIMITER_BUILD_BENCHMARKS="$benchmarks" "${dataset_option[@]}"
cmake --build "$build_dir" --parallel 2
ctest --test-dir "$build_dir" --output-on-failure

if [ "$configuration" = Release ]; then
  export PHASELIMITER_BIN_DIR="$PWD/$build_dir/bin"
  bundle exec rubocop --lint
  bundle exec rspec
  python3 -B -m unittest discover -s tests/build -v
  mkdir -p build/data build/artifacts
  "$build_dir/bin/audio_analyzer" --mode=sound_quality2_preparation \
    --analysis_data_dir="$PWD/build/dataset/analysis_shortpath" \
    --sound_quality2_cache="$PWD/build/data/sound_quality2_cache"
  python3 script/ci_checks.py --mode integration --bin-dir "$build_dir/bin" \
    --output-dir build/results/integration --sound-quality-cache build/data/sound_quality2_cache
  python3 script/stage_release.py --build-dir "$build_dir" --destination build/stage/linux \
    --sound-quality-cache build/data/sound_quality2_cache
  python3 script/package_release.py --stage-dir build/stage/linux --output build/artifacts/release.tar.xz
fi
# Verify that configuring, testing and packaging did not change source files.
test -z "$(git status --porcelain)"
git submodule foreach --quiet --recursive 'test -z "$(git status --porcelain)"'
