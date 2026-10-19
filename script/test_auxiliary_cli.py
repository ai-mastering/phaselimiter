#!/usr/bin/env python3
"""Exercise auxiliary CLI tools and validate their media outputs."""
import argparse
import json
import math
from pathlib import Path
import struct
import subprocess
import tempfile
import wave


def check_tools(bin_dir, output, valgrind=False):
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    # Each invocation gets fresh outputs, retained for debugging failures.
    work = Path(tempfile.mkdtemp(prefix='auxiliary-', dir=output))
    source = repo / 'test_data/test5.wav'
    with wave.open(str(source), 'rb') as audio:
        channels, rate, frames = audio.getnchannels(), audio.getframerate(), audio.getnframes()

    def run(name, command, instrument=False):
        command = [str(part) for part in command]
        if valgrind and instrument:
            command = ['valgrind', '--error-exitcode=1', '--leak-check=full',
                       '--show-leak-kinds=all', '--errors-for-leak-kinds=definite,indirect',
                       '--track-origins=yes', f'--log-file={work / (name + ".valgrind.log")}',
                       *command]
        print(f'{name}: {command}', flush=True)
        with (work / (name + '.stdout')).open('wb') as stdout, \
                (work / (name + '.stderr')).open('wb') as stderr:
            result = subprocess.run(command, cwd=work, stdout=stdout, stderr=stderr)
        if result.returncode:
            print((work / (name + '.stderr')).read_text(errors='replace'), flush=True)
            if valgrind and instrument:
                print((work / (name + '.valgrind.log')).read_text(errors='replace'), flush=True)
            result.check_returncode()
        return (work / (name + '.stdout')).read_bytes()

    image_dir = work / 'visualizer'
    image_dir.mkdir()
    run('visualizer', [bin_dir / 'audio_visualizer', f'--input={source}',
        f'--output={image_dir}', f'--background={repo / "test_data/icon_blue.jpg"}',
        f'--foreground={repo / "test_data/icon_blue.png"}',
        '--width=640', '--height=360', '--fps=25'], instrument=True)
    images = list(image_dir.glob('*.png'))
    expected_frames = math.ceil(frames * 25 / rate)
    if len(images) != expected_frames or {p.stem for p in images} != {str(i) for i in range(expected_frames)}:
        raise AssertionError(f'Missing visualizer frame sequence: {image_dir}')
    for image in images:
        header = image.read_bytes()[:24]
        if header[:16] != b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR' or \
                struct.unpack('>II', header[16:24]) != (640, 360):
            raise AssertionError(f'Invalid PNG or frame dimensions: {image}')
    # Decode every frame as well, to reject corrupt/truncated PNG payloads.
    run('decode_visualizer', ['ffmpeg', '-v', 'error', '-xerror', '-i',
        image_dir / '%d.png', '-f', 'null', '-'])

    spectrum = json.loads(run('spectrogram', [bin_dir / 'spectrogram',
        f'--input={source}'], instrument=True))
    rows = spectrum['spectrogram']
    if (spectrum['channels'], spectrum['sample_rate'], spectrum['frames']) != (channels, rate, frames):
        raise AssertionError('Spectrogram metadata differs from input')
    if not rows or not rows[0] or any(len(row) != len(rows[0]) for row in rows):
        raise AssertionError('Spectrogram must be a nonempty rectangular matrix')
    if not all(math.isfinite(value) and value >= 0 for row in rows for value in row):
        raise AssertionError('Spectrogram contains invalid energy values')
    if not any(value > 0 for row in rows for value in row):
        raise AssertionError('Spectrogram contains no signal')

    for name, inputs, flags in [
            ('clear_mixer', f'{source},{source}', []),
            ('effect_transient', str(source), ['--mode=transient']),
            ('effect_transient2', str(source), ['--mode=transient2'])]:
        tool = 'clear_mixer' if name == 'clear_mixer' else 'effect_test'
        target = work / (name + '.wav')
        run(name, [bin_dir / tool, f'--input={inputs}', f'--output={target}', *flags], instrument=True)
        metadata = json.loads(run(name + '_info', [bin_dir / 'snd_file_info', f'--input={target}']))
        if tuple(int(metadata[key]) for key in ('channels', 'samplerate', 'frames')) != (channels, rate, frames):
            raise AssertionError(f'{name}: output audio dimensions differ from input')
        decoded = run(name + '_decode', ['ffmpeg', '-v', 'error', '-xerror',
            '-i', target, '-f', 'f32le', '-acodec', 'pcm_f32le', '-'])
        if len(decoded) != frames * channels * 4:
            raise AssertionError(f'{name}: decoded audio length differs from input')
        samples = [value for (value,) in struct.iter_unpack('<f', decoded)]
        if not all(math.isfinite(value) for value in samples) or not any(value != 0 for value in samples):
            raise AssertionError(f'{name}: output contains invalid samples or no signal')
    print(f'Auxiliary CLI checks passed; outputs: {work}', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--valgrind', action='store_true')
    args = parser.parse_args()
    check_tools(args.bin_dir.resolve(), args.output_dir.resolve(), args.valgrind)


if __name__ == '__main__':
    main()
