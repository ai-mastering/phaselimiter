#!/usr/bin/env python3
"""Native detect2 CLI contracts; standard library only, no audio DSP or Python ML."""
import argparse
import json
import math
from pathlib import Path
import subprocess
import tempfile


def analysis(mean):
    return {'loudness': -10, 'bands_short': [
        {'mid_mean': mean - 10, 'side_mean': mean - 10}],
        'covariance_short': [[1, 0], [0, 1]]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    tool = parser.parse_args().tool.resolve()
    with tempfile.TemporaryDirectory(prefix='audio-detector-') as temporary:
        root = Path(temporary)
        data = root / 'resource' / 'analysis_data'
        data.mkdir(parents=True)
        unrelated = root / 'unrelated'
        unrelated.mkdir()
        paths = []
        for i, mean in enumerate([0, 1, 2, 10]):
            name = f'{i:06d}.json'
            (data / name).write_text(json.dumps(analysis(mean)))
            paths.append('./resource/analysis_data/' + name)
        preparation = root / 'resource' / 'detection_preparation.json'
        preparation.write_text(json.dumps({
            'paths': paths, 'covariance_regularization': .001,
            'distance_matrix': [[0, 1, 2, 10], [1, 0, 1, 9],
                                [2, 1, 0, 8], [10, 9, 8, 0]]}))
        source = root / 'input analysis.json'
        source.write_text(json.dumps(analysis(1)))
        flags = ['--source', str(source), '--detection_preparation', str(preparation)]

        def run(*arguments, success=True):
            result = subprocess.run([str(tool), *arguments], cwd=unrelated,
                                    capture_output=True, text=True)
            if success:
                assert result.returncode == 0, result.stderr
                return json.loads(result.stdout)
            assert result.returncode != 0
            assert not result.stdout.strip(), result.stdout
            assert result.stderr.strip()

        result = run('detect2', *flags, '--neighbor_ratio', '.5', '--outlier_ratio', '.1')
        border = (119 / 24) * .7 + (4 / 3) * .3
        assert math.isclose(result['lof'], .875, abs_tol=1e-9), result
        assert math.isclose(result['border_lof'], border, abs_tol=1e-9), result
        assert result['ranking'][0]['path'] == paths[1]
        assert math.isclose(result['ranking'][0]['lof'], 4 / 3, abs_tol=1e-9)
        assert math.isclose(result['sound_quality'], 1 / (1 + .125 / (border - 1)), abs_tol=1e-9)
        assert run('--task=detect2', *flags) == run('detect2', *flags, '--neighbor_ratio=.03')
        run('categorize', *flags, success=False)
        run('prepare_for_detection', success=False)
        run('detect2', success=False)
        run('detect2', *flags, '--neighbor_ratio=0', success=False)
        source.write_text('{"loudness": 0}')
        run('detect2', *flags, success=False)
        # Preparation-relative anonymous paths work without any resource/ prefix.
        doc = json.loads(preparation.read_text())
        doc['paths'] = ['./analysis_data/' + Path(p).name for p in paths]
        preparation.write_text(json.dumps(doc))
        source.write_text(json.dumps(analysis(1)))
        assert run('detect2', *flags)['ranking'][0]['path'] == doc['paths'][1]
    print('audio_detector CLI contracts passed')


if __name__ == '__main__':
    main()
