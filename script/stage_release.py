#!/usr/bin/env python3
"""Install tools and release data into an empty staging directory."""
import argparse
from pathlib import Path
import shutil
import subprocess


def stage(build_dir, destination, config, cache=None):
    repo = Path(__file__).resolve().parents[1]
    destination = destination.resolve()
    if destination.is_relative_to(repo) and not destination.is_relative_to(repo / 'build'):
        raise ValueError('Stage under build/ or outside the source repository')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError(f'Staging directory must be empty: {destination}')
    prefix = destination / 'phaselimiter'
    subprocess.run(['cmake', '--install', str(build_dir), '--config', config,
                    '--prefix', str(prefix)], check=True)
    manifest = repo / 'packaging/release_files.txt'
    for entry in manifest.read_text().splitlines():
        relative = Path(entry).relative_to('phaselimiter')
        target = prefix / relative
        if str(relative) in ('bin', 'lib'):
            target.mkdir(parents=True, exist_ok=True)
            continue
        source = repo / ('packaging/licenses' if str(relative) == 'licenses' else relative)
        if source.is_dir():
            shutil.copytree(source, target,
                            ignore=shutil.ignore_patterns('.DS_Store', 'sound_quality2_cache'))
        else:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    if cache is not None:
        shutil.copy2(cache, prefix / 'resource/sound_quality2_cache')
    return prefix


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--build-dir', type=Path, required=True)
    parser.add_argument('--destination', type=Path, required=True)
    parser.add_argument('--config', default='Release')
    parser.add_argument('--sound-quality-cache', type=Path)
    args = parser.parse_args()
    stage(args.build_dir.resolve(), args.destination, args.config, args.sound_quality_cache)


if __name__ == '__main__':
    main()
