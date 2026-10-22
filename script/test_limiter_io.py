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

        # Missing input and FFmpeg prove these fail before opening/converting audio.
        invalid_cases = [
            (['--worker_count=-3'], 'worker_count'),
            (['--test_mode=typo'], 'test_mode'),
            (['--input='], 'input'),
            (['--output='], 'output'),
            (['--output_format=flac'], 'output_format'),
            (['--reference_mode=typo'], 'reference_mode'),
            (['--limiting_mode=typo'], 'limiting_mode'),
            (['--ceiling_mode=typo'], 'ceiling_mode'),
            (['--noise_update_mode=typo'], 'noise_update_mode'),
            (['--max_available_freq_mode=typo'], 'max_available_freq_mode'),
            (['--sample_rate=0'], 'sample_rate'),
            (['--bit_depth=8'], 'bit_depth'),
            (['--true_peak_oversample=0'], 'true_peak_oversample'),
            (['--true_peak_oversample=2147483647'], 'true_peak_oversample'),
            (['--reference=nan'], 'reference must be finite'),
            (['--ceiling=inf'], 'ceiling must be finite'),
            (['--low_cut_freq=-1'], 'low_cut_freq'),
            (['--high_cut_freq=22050'], 'high_cut_freq'),
            (['--low_cut_freq=1000', '--high_cut_freq=500'], 'below'),
            (['--ceiling_mode=lowpass_true_peak', '--lowpass_true_peak_cut_freq=0'],
             'lowpass_true_peak_cut_freq'),
            (['--noise_update_min_noise=0'], 'noise levels'),
            (['--absolute_min_noise=-1'], 'noise levels'),
            (['--max_iter1=0'], 'max_iter1'),
            (['--max_iter2=-1'], 'max_iter2'),
            (['--pre_compression=true', '--pre_compression_mean_sec=0'], 'pre_compression_mean_sec'),
            (['--mastering', '--mastering_mode=typo'], 'mastering_mode'),
            (['--mastering', '--mastering_mode=mastering3', '--mastering3_iteration=0'],
             'mastering3_iteration'),
            (['--mastering', '--mastering_mode=mastering5',
              '--mastering5_optimization_algorithm=typo'], 'mastering5_optimization_algorithm'),
            (['--mastering', '--mastering_mode=mastering5',
              '--mastering5_optimization_max_eval_count=0'], 'mastering5_optimization_max_eval_count'),
        ]
        for flags, message in invalid_cases:
            target.write_bytes(b'original')
            intermediate = root / 'preserved.wav'
            intermediate.write_bytes(b'original intermediate')
            scratch = root / 'unused temporary'
            command = base + [f'--input={root / "missing.wav"}',
                              f'--ffmpeg={root / "missing-ffmpeg"}',
                              f'--tmp={scratch}',
                              f'--output_after_pre_compression={intermediate}'] + flags
            result = subprocess.run(command, capture_output=True, text=True, timeout=10)
            assert result.returncode != 0, flags
            assert message in result.stderr, (flags, result.stderr)
            assert 'Ipp initialized' not in result.stderr, result.stderr
            assert not scratch.exists(), flags
            assert target.read_bytes() == b'original'
            assert intermediate.read_bytes() == b'original intermediate'

        help_result = subprocess.run([str(tool), '--helpfull'],
                                     capture_output=True, text=True, timeout=10)
        help_text = help_result.stdout + help_result.stderr
        assert 'Output formats: WAV' in help_text, help_text
        assert 'MP3, AAC' in help_text, help_text
        assert '32-bit float' in help_text, help_text
        assert 'default: true' in help_text.split('-quick_exit', 1)[1].split('\n', 1)[0], help_text

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
        # Exercise every documented format, including float WAV and rate conversion.
        for output_format, bit_depth, rate in [('wav', 16, 44100), ('wav', 24, 48000),
                                               ('wav', 32, 44100), ('mp3', 16, 44100),
                                               ('aac', 16, 44100)]:
            output = root / f'format-{output_format}-{bit_depth}.{output_format}'
            result = subprocess.run(
                base + [f'--input={source}', f'--output={output}',
                        f'--output_format={output_format}', f'--bit_depth={bit_depth}',
                        f'--sample_rate={rate}'],
                capture_output=True, text=True, timeout=60)
            assert result.returncode == 0, result.stderr
            decoded = root / 'decoded.wav'
            subprocess.run([ffmpeg, '-v', 'error', '-y', '-i', str(output),
                            '-acodec', 'pcm_s16le', str(decoded)],
                           check=True, capture_output=True, timeout=60)
            with wave.open(str(decoded), 'rb') as audio:
                assert audio.getnchannels() == 2
                assert audio.getframerate() == rate
                assert audio.getnframes() > 0
    print('Limiter input validation and FFmpeg path integration passed')


if __name__ == '__main__':
    main()
