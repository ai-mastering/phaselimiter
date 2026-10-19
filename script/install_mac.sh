#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$repo_dir/build/macos"
cmake -S "$repo_dir" -B "$build_dir" -G Xcode -DBUILD_TESTING=OFF
cmake --build "$build_dir" --config Release --parallel 2

# Keep the legacy application/data location without copying build products
# and third-party source trees into /etc/phaselimiter.
sudo mkdir -p /etc/phaselimiter
for entry in resource script test_data .python-version LICENSE README.md Pipfile Pipfile.lock; do
  sudo cp -R "$repo_dir/$entry" /etc/phaselimiter/
done
sudo cp -R "$repo_dir/packaging/licenses" /etc/phaselimiter/
sudo cmake --install "$build_dir" --config Release --prefix /etc/phaselimiter
sudo cmake --install "$build_dir" --config Release --prefix /usr/local
sudo install -m 755 "$repo_dir/script/audio_detector" /usr/local/bin/audio_detector
cd /etc/phaselimiter
pyenv exec pipenv install
