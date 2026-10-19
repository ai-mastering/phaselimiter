#!/usr/bin/env python3
"""Check WAV reader contracts for empty/short, mono/stereo, and 44.1/48 kHz audio."""
import argparse
import itertools
import json
from pathlib import Path
import subprocess
import tempfile
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    args = parser.parse_args()
    tool = args.tool.resolve()
    with tempfile.TemporaryDirectory(prefix='phaselimiter-audio-') as directory:
        path = Path(directory) / 'input.wav'
        for frames, channels, rate in itertools.product(
                [0, 1, 7, 8, 9, 255, 256, 257], [1, 2], [44100, 48000]):
            with wave.open(str(path), 'wb') as output:
                output.setnchannels(channels)
                output.setsampwidth(2)
                output.setframerate(rate)
                output.writeframes(b'\x00\x00' * frames * channels)
            result = subprocess.run([str(tool), f'--input={path}'],
                                    capture_output=True, text=True, timeout=30)
            case = f'{frames} frames, {channels} channels, {rate} Hz'
            if result.returncode:
                raise RuntimeError(f'{case}: exit {result.returncode}\n{result.stderr}')
            metadata = json.loads(result.stdout)
            for key, expected in [('frames', frames), ('channels', channels), ('samplerate', rate)]:
                if int(metadata[key]) != expected:
                    raise AssertionError(f'{case}: {key}={metadata[key]}, expected {expected}')
    print('Passed 32 WAV boundary cases')


if __name__ == '__main__':
    main()
