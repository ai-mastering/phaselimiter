#!/bin/bash
set -euo pipefail
repo_dir="$(cd "$(dirname "$0")/.." && pwd)"
docker build --platform linux/amd64 --target development -t phaselimiter-development "$repo_dir"
docker run --platform linux/amd64 -it --rm --mount "type=bind,src=$repo_dir,dst=/workspace" \
  --workdir /workspace phaselimiter-development bash
