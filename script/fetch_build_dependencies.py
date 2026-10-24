#!/usr/bin/env python3
"""Fetch the SDKs pinned by environment/dependencies.json."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile
import urllib.request
import zipfile


def fetch(asset, cache):
    archive = cache / asset['url'].rsplit('/', 1)[-1]
    if not archive.exists():
        temporary = archive.with_suffix(archive.suffix + '.part')
        try:
            with urllib.request.urlopen(asset['url'], timeout=120) as response:
                with temporary.open('wb') as output:
                    shutil.copyfileobj(response, output)
            temporary.replace(archive)
        finally:
            temporary.unlink(missing_ok=True)
    digest = hashlib.sha256()
    with archive.open('rb') as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b''):
            digest.update(chunk)
    if digest.hexdigest() != asset['sha256']:
        raise ValueError(f'Checksum mismatch: {archive}; remove it before retrying')
    return archive


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--platform', choices=['linux', 'windows', 'macos-intel'], required=True)
    parser.add_argument('--prefix', type=Path, required=True)
    parser.add_argument('--cache-dir', type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'environment/dependencies.json'
    assets = json.loads(manifest.read_text())[args.platform]
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    for asset in assets:
        archive = fetch(asset, args.cache_dir)
        destination = args.prefix.resolve() / asset['destination']
        destination.mkdir(parents=True, exist_ok=True)
        if asset['format'] == 'deb':
            subprocess.run(['dpkg-deb', '--extract', str(archive), str(destination)], check=True)
        elif asset['format'] == 'zip':
            with zipfile.ZipFile(archive) as package:
                package.extractall(destination)
        else:
            with tarfile.open(archive) as package:
                package.extractall(destination, filter='data')
        print(f"Installed {asset['name']} {asset['version']}: {destination}", flush=True)


if __name__ == '__main__':
    main()
