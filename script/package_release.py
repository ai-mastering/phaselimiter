#!/usr/bin/env python3
"""Archive an existing release stage without modifying it."""
import argparse
from pathlib import Path
import tarfile
import zipfile


def package(stage_dir, output):
    if output.resolve().is_relative_to(stage_dir.resolve()):
        raise ValueError('Archive must be outside the staging directory')
    root = stage_dir / 'phaselimiter'
    if not (root / 'bin').is_dir():
        raise ValueError(f'Missing installed tools: {root / "bin"}')
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix == '.zip':
        with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(root.rglob('*')):
                archive.write(path, path.relative_to(stage_dir))
    elif output.name.endswith('.tar.xz'):
        with tarfile.open(output, 'w:xz') as archive:
            archive.add(root, arcname='phaselimiter')
    else:
        raise ValueError('Output must end in .zip or .tar.xz')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    package(args.stage_dir.resolve(), args.output.resolve())


if __name__ == '__main__':
    main()
