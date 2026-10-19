#!/bin/bash
set -euo pipefail

repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
build_dir="$repo_dir/build/macos"
cmake -S "$repo_dir" -B "$build_dir" -G Xcode -DBUILD_TESTING=OFF
cmake --build "$build_dir" --config Release --parallel 2

# Keep the legacy application/data location without copying build products
# and third-party source trees into /etc/phaselimiter.
sudo mkdir -p /etc/phaselimiter
for entry in resource LICENSE README.md; do
  sudo cp -R "$repo_dir/$entry" /etc/phaselimiter/
done
sudo cp -R "$repo_dir/packaging/licenses" /etc/phaselimiter/
sudo cmake --install "$build_dir" --config Release --prefix /etc/phaselimiter
sudo cmake --install "$build_dir" --config Release --prefix /usr/local
