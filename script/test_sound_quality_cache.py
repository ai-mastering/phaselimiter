#!/usr/bin/env python3
"""Check model reproducibility across filesystem traversal orders and worker counts."""
import argparse
import json
from pathlib import Path
import random
import subprocess
import tempfile


def check(tool):
    rng = random.Random(1)
    references = {}
    for i in range(32):
        covariance = [[0.0] * 4 for _ in range(4)]
        for row in range(4):
            covariance[row][row] = 2 + rng.random()
            for column in range(row):
                covariance[row][column] = covariance[column][row] = .1 * rng.random()
        bands = [dict(low_freq=low, high_freq=high, mid_mean=-20 + 8 * rng.random(),
                      side_mean=-25 + 8 * rng.random())
                 for low, high in [(0, 1000), (1000, 22050)]]
        references[f'{i:03}.json'] = json.dumps(dict(bands=bands, covariance=covariance))

    with tempfile.TemporaryDirectory(prefix='phaselimiter-cache-repeatability-') as directory:
        root = Path(directory)
        expected = None
        for label, nested, workers in [('flat', False, 1), ('nested', True, 1), ('parallel', True, 2)]:
            source = root / label
            source.mkdir()
            # Flat/nested layouts preserve lexical record order, but change
            # filesystem hash traversal even on filesystems insensitive to
            # creation order. This exercises the enumeration-order regression.
            for name in sorted(references, reverse=nested):
                target = source / name
                if nested:
                    target = source / name.removesuffix('.json') / 'record.json'
                    target.parent.mkdir()
                target.write_text(references[name])
            cache = root / (label + '.cache')
            result = subprocess.run([str(tool), '--mode=sound_quality2_preparation',
                                     f'--analysis_data_dir={source}', f'--sound_quality2_cache={cache}',
                                     f'--worker_count={workers}'], cwd=root, capture_output=True,
                                    text=True, timeout=120)
            if result.returncode:
                raise RuntimeError(f'{label} cache preparation failed:\n{result.stderr}')
            actual = cache.read_bytes()
            if not actual:
                raise ValueError('Generated cache is empty')
            if expected is None:
                expected = actual
            elif actual != expected:
                raise ValueError(f'{label}: cache differs with filesystem traversal order or worker count')
    print('Cache bytes match across filesystem traversal orders and worker counts')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    args = parser.parse_args()
    check(args.tool.resolve())


if __name__ == '__main__':
    main()
