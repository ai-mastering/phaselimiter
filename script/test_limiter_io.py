#!/usr/bin/env python3
"""Check limiter input validation and literal FFmpeg paths through the CLI."""
import argparse
import math
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import wave


def write_wave(path, channels, rate):
    samples = [int(4000 * math.sin(2 * math.pi * 440 * i / rate))
               for i in range(rate // 4)]
    with wave.open(str(path), 'wb') as output:
        output.setnchannels(channels)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes(b''.join(struct.pack('<h', sample) * channels for sample in samples))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tool', type=Path, required=True)
    args = parser.parse_args()
    tool = args.tool.resolve()
    with tempfile.TemporaryDirectory(prefix='phaselimiter-io-') as directory:
        root = Path(directory)
        source = root / 'input.wav'
        target = root / 'output.wav'
        base = [str(tool), f'--input={source}', f'--output={target}',
                f'--tmp={root / "temporary"}', '--reference_mode=peak',
                '--reference=-6', '--pre_compression=false']
        for channels, rate in [(1, 44100), (1, 48000), (2, 48000)]:
            write_wave(source, channels, rate)
            target.write_bytes(b'original')
            result = subprocess.run(base + ['--disable_input_encode'],
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode != 0, result.stdout
            assert 'Input audio must have 2 channels at 44100 Hz' in result.stderr, result.stderr
            assert target.read_bytes() == b'original'

        write_wave(source, 2, 44100)
        for flags, message in [
            (['--limiter_external_oversample=0'], 'positive powers of 2'),
            (['--limiter_internal_oversample=-1'], 'positive powers of 2'),
            (['--limiter_external_oversample=1073741824',
              '--limiter_internal_oversample=1073741824'], 'sample rate exceeds'),
            (['--enhancement'], 'require FFTW support'),
            (['--freq_expansion'], 'require FFTW support'),
            (['--start_at=nan'], 'must be finite'),
            (['--end_at=inf'], 'must be finite'),
        ]:
            target.write_bytes(b'original')
            result = subprocess.run(base + ['--disable_input_encode'] + flags,
                                    capture_output=True, text=True, timeout=60)
            assert result.returncode != 0, result.stdout
            assert message in result.stderr, result.stderr
            assert target.read_bytes() == b'original'

        ffmpeg = shutil.which('ffmpeg')
        if not ffmpeg:
            print('Input rejection cases passed; FFmpeg integration skipped (not installed)')
            return
        # The executable path and both audio paths contain shell-sensitive characters.
        executable = root / ('ffmpeg space $ & tool' + Path(ffmpeg).suffix)
        shutil.copy2(ffmpeg, executable)
        source = root / 'input space $ & (audio).wav'
        target = root / 'output space $ & (audio).wav'
        intermediate = root / 'pre compression $ &.wav'
        for channels, rate, disable_encode in [(2, 44100, True), (1, 48000, False)]:
            write_wave(source, channels, rate)
            target.write_bytes(b'original')
            intermediate.write_bytes(b'original intermediate')
            command = [str(tool), f'--input={source}', f'--output={target}',
                       f'--output_after_pre_compression={intermediate}',
                       f'--tmp={root / "temporary"}', f'--ffmpeg={executable}',
                       '--reference_mode=peak', '--reference=-6', '--pre_compression=false']
            if disable_encode:
                command.extend(['--disable_input_encode', '--start_at=0.05', '--end_at=0.15'])
            result = subprocess.run(command, capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            for output in [target, intermediate]:
                with wave.open(str(output), 'rb') as audio:
                    assert audio.getnchannels() == 2
                    assert audio.getframerate() == 44100
                    assert audio.getnframes() > 0
                    if disable_encode and output == target:
                        assert audio.getnframes() == 4410, audio.getnframes()
    print('Limiter input validation and FFmpeg path integration passed')


if __name__ == '__main__':
    main()
