#!/bin/bash
set -euo pipefail

archive="$(realpath "${1:?Usage: install_linux.sh /path/to/release.tar.xz}")"
install_tmpdir="$(mktemp -d)"
trap 'rm -rf "$install_tmpdir"' EXIT

tar -Jxf "$archive" -C "$install_tmpdir"
release_dir="$install_tmpdir/phaselimiter"
if [ ! -x "$release_dir/bin/audio_detector" ]; then
  echo "Release archive must contain the native bin/audio_detector executable" >&2
  exit 1
fi

sudo mkdir -p /etc/phaselimiter /usr/local/bin
sudo cp -R "$release_dir/." /etc/phaselimiter/
sudo cp "$release_dir/bin/"* /usr/local/bin/
