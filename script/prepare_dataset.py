#!/usr/bin/env python3
"""Check out the fixed dataset revision into the build tree."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--destination', type=Path, required=True)
    args = parser.parse_args()
    manifest = Path(__file__).resolve().parents[1] / 'environment/dataset.json'
    dataset = json.loads(manifest.read_text())
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not (destination / '.git').exists():
        subprocess.run(['git', 'init', str(destination)], check=True)
    subprocess.run(['git', '-C', str(destination), 'fetch', '--depth=1',
                    dataset['url'], dataset['revision']], check=True)
    subprocess.run(['git', '-C', str(destination), 'checkout', '--detach', 'FETCH_HEAD'], check=True)


if __name__ == '__main__':
    main()
