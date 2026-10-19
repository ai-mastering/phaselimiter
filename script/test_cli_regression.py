#!/usr/bin/env python3
"""CLI regression coverage migrated from the former RSpec tests."""
import argparse
import itertools
import json
from pathlib import Path
import shlex
import shutil
import struct
import subprocess
import sys
import tempfile
import unittest


REPO = Path(__file__).resolve().parents[1]


class CliTestCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory(prefix='phaselimiter-regression-')
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name)

    def run_command(self, command):
        command = [str(value) for value in command]
        result = subprocess.run(command, cwd=self.work, capture_output=True,
                                text=True, timeout=300)
        self.assertEqual(result.returncode, 0,
                         f'{shlex.join(command)}\n{result.stdout}\n{result.stderr}')
        return result.stdout

    def tool(self, name):
        path = self.bin_dir / name
        if sys.platform == 'win32':
            path = path.with_suffix('.exe')
        self.assertTrue(path.is_file(), f'Missing executable: {path}')
        return path

    def limit(self, source, output, *flags):
        self.run_command([
            self.tool('phase_limiter'), f'--input={source}', f'--output={output}',
            f'--tmp={self.work / "native"}',
            f'--mastering_reference_file={REPO / "resource/mastering_reference.json"}',
            f'--mastering_reverb_ir_left={REPO / "test_data/test5.wav"}',
            f'--mastering_reverb_ir_right={REPO / "test_data/test5.wav"}',
            '--mastering_reverb_gain=0', '--mastering_reverb_predelay=0.04',
            '--mastering_reverb_drr_range=30', *flags,
        ])
        self.assertTrue(output.is_file(), f'Missing output: {output}')
        self.assertGreater(output.stat().st_size, 0)


@unittest.skipUnless(shutil.which('ffmpeg'), 'requires FFmpeg')
class LimiterRegression(CliTestCase):
    def test_smoke(self):
        cases = [
            ('peak', 'test2.wav', 'wav',
             ['--reference_mode=peak', '--reference=0', '--pre_compression=false']),
            ('limiting', 'test2.wav', 'wav', []),
            ('short', 'test5.wav', 'wav', []),
            ('24bit', 'test2.wav', 'wav', ['--bit_depth=24']),
            ('mp3', 'test1.mp3', 'mp3', ['--output_format=mp3']),
            ('many_workers', 'test2.wav', 'wav', ['--worker_count=64']),
            ('mastering', 'test2.wav', 'wav', ['--mastering=true']),
            ('reverb', 'test2.wav', 'wav', ['--mastering=true', '--mastering_reverb=true']),
            ('range', 'test2.wav', 'wav',
             ['--mastering=true', '--start_at=0.7', '--end_at=1.7']),
            ('mastering3', 'test2.wav', 'wav',
             ['--mastering=true', '--mastering_mode=mastering3']),
        ]
        for name, source, extension, flags in cases:
            with self.subTest(case=name):
                self.limit(REPO / 'test_data' / source,
                           self.work / f'{name}.{extension}', *flags)

    def test_preserves_decoded_frame_counts(self):
        formats = [('aac', 'aac'), ('aac', 'wav'), ('mp3', 'mp3'),
                   ('mp3', 'wav'), ('wav', 'wav')]
        # Compare decoded input/output lengths: compressed inputs already include
        # codec padding. The original suite covered 44.1 kHz only.
        for (input_format, output_format), frames in itertools.product(
                formats, [2000, 44100, 56789]):
            with self.subTest(input=input_format, output=output_format, frames=frames):
                directory = self.work / f'{input_format}-{output_format}-{frames}'
                directory.mkdir()
                raw = directory / 'impulse.raw'
                raw.write_bytes(struct.pack('<h', 32767) + b'\0\0' * (frames - 1))
                source = directory / f'input.{input_format}'
                target = directory / f'output.{output_format}'
                self.run_command(['ffmpeg', '-v', 'error', '-nostdin', '-y',
                                  '-f', 's16le', '-ar', 44100, '-ac', 1,
                                  '-i', raw, source])
                self.limit(source, target, f'--output_format={output_format}',
                           '--sample_rate=44100')
                lengths = []
                for name, path in [('input', source), ('output', target)]:
                    decoded = directory / f'{name}_decoded.wav'
                    self.run_command(['ffmpeg', '-v', 'error', '-nostdin', '-y',
                                      '-i', path, decoded])
                    info = json.loads(self.run_command([
                        self.tool('snd_file_info'), f'--input={decoded}']))
                    lengths.append(int(info['frames']))
                self.assertEqual(lengths[0], lengths[1])


class AudioAnalyzerRegression(CliTestCase):
    def test_analysis_values(self):
        result = json.loads(self.run_command([
            self.tool('audio_analyzer'), f'--tmp={self.work / "native"}',
            f'--input={REPO / "test_data/test2.wav"}',
            f'--analysis_data_dir={REPO / "resource/analysis_data"}',
            '--sound_quality2=false',
        ]))
        expected = {'channels': 2, 'format': 65538, 'frames': 280049,
                    'sample_rate': 44100, 'sections': 1, 'seekable': 1}
        for key, value in expected.items():
            with self.subTest(field=key):
                self.assertEqual(result[key], value)
        expected_values = {'peak': -3.77, 'rms': -16.3, 'loudness': -14.8,
                           'loudness_range': 2.0, 'dynamics': 1.39,
                           'sharpness': 2.19, 'space': 0.325}
        for key, value in expected_values.items():
            with self.subTest(field=key):
                self.assertAlmostEqual(result[key], value, delta=0.1)
        for key in ['spectrum', 'waveform', 'histogram', 'loudness_time_series',
                    'freq_pan_to_db']:
            with self.subTest(field=key):
                self.assertIsNotNone(result.get(key))
                self.assertIsNot(result[key], False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    args, test_args = parser.parse_known_args()
    CliTestCase.bin_dir = args.bin_dir.resolve()
    unittest.main(argv=[sys.argv[0], *test_args])
