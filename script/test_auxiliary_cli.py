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
import zlib


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

    def write_png(path, width, height, depth, color_type, rows):
        def chunk(kind, data):
            return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
        path.write_bytes(b'\x89PNG\r\n\x1a\n' +
            chunk(b'IHDR', struct.pack('>IIBBBBB', width, height, depth, color_type, 0, 0, 0)) +
            chunk(b'IDAT', zlib.compress(b''.join(b'\0' + row for row in rows))) + chunk(b'IEND', b''))

    foreground = work / 'foreground.png'
    write_png(foreground, 65, 49, 8, 6, [bytes(
        value for x in range(65) for value in
        ((x * 13) % 256, (y * 17) % 256, (x * 7 + y * 11) % 256, (0, 90, 200, 255)[x % 4]))
        for y in range(49)])
    image_dir = work / 'visualizer'
    image_dir.mkdir()
    visualizer_flags = [f'--input={source}', f'--background={repo / "test_data/icon_blue.jpg"}',
        f'--foreground={foreground}', '--width=640', '--height=360', '--fps=25', '--worker_count=8']
    run('visualizer', [bin_dir / 'audio_visualizer', f'--output={image_dir}',
        *visualizer_flags], instrument=True)
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

    def decode_images(data, codec):
        return subprocess.run(['ffmpeg', '-v', 'error', '-xerror', '-threads', '8',
            '-filter_threads', '8', '-f', 'image2pipe', '-c:v', codec, '-framerate', '25', '-i', '-',
            '-f', 'rawvideo', '-pix_fmt', 'rgb48le', '-'], input=data,
            stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True).stdout

    stream = run('visualizer_ppm', [bin_dir / 'audio_visualizer', '--output_format=ppm',
        *visualizer_flags], instrument=True)
    png_stream = b''.join((image_dir / f'{i}.png').read_bytes() for i in range(expected_frames))
    # The legacy --output=- invocation must still produce the original PNG stream.
    legacy_stream = run('visualizer_png_stdout', [bin_dir / 'audio_visualizer', '--output=-',
        *visualizer_flags])
    explicit_png = run('visualizer_png_explicit', [bin_dir / 'audio_visualizer',
        '--output_format=png', *visualizer_flags])
    if legacy_stream != png_stream or explicit_png != png_stream:
        raise AssertionError('Default or explicit PNG stdout differs from directory PNG frames')
    ppm_dir = work / 'visualizer_ppm'
    ppm_dir.mkdir()
    run('visualizer_ppm_directory', [bin_dir / 'audio_visualizer', '--output_format=ppm',
        f'--output={ppm_dir}', *visualizer_flags])
    if len(list(ppm_dir.iterdir())) != expected_frames or stream != b''.join(
            (ppm_dir / f'{i}.ppm').read_bytes() for i in range(expected_frames)):
        raise AssertionError('PPM directory frames differ from PPM stdout')
    invalid = subprocess.run([str(bin_dir / 'audio_visualizer'), '--output_format=jpeg'],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if invalid.returncode == 0 or invalid.stdout or b'output_format' not in invalid.stderr:
        raise AssertionError('Unsupported output format was not rejected clearly')
    decoded_ppm = decode_images(stream, 'ppm')
    if not stream.startswith(b'P6\n640 360\n255\n') or \
            len(decoded_ppm) != expected_frames * 640 * 360 * 6 or \
            decoded_ppm != decode_images(png_stream, 'png'):
        raise AssertionError('PPM stdout differs from PNG frame pixels or ordering')
    serial = run('visualizer_ppm_serial', [bin_dir / 'audio_visualizer', '--output=-', '--output_format=ppm',
        *visualizer_flags, '--worker_count=1'])
    if serial != stream:
        raise AssertionError('PPM stdout depends on worker count')
    # Exercise the real PPM-to-H.264 pipe contract used by callers.
    subprocess.run(['ffmpeg', '-nostdin', '-v', 'error', '-xerror', '-threads', '8',
        '-filter_threads', '8', '-f', 'image2pipe', '-c:v', 'ppm', '-framerate', '25', '-i', '-',
        '-an', '-c:v', 'libx264', '-threads', '8', '-crf', '23', '-maxrate', '4M', '-bufsize', '8M',
        '-pix_fmt', 'yuv420p', '-f', 'mp4', '-y', str(work / 'visualizer.mp4')],
        input=stream, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)

    background16 = work / 'background16.png'
    write_png(background16, 1, 1, 16, 2, [struct.pack('>3H', 1024, 2048, 4096)])
    image_dir16 = work / 'visualizer16'
    image_dir16.mkdir()
    flags16 = [f'--input={source}', f'--background={background16}',
        '--width=17', '--height=13', '--fps=25', '--worker_count=8']
    run('visualizer_png16', [bin_dir / 'audio_visualizer', f'--output={image_dir16}', *flags16])
    stream16 = run('visualizer_ppm16', [bin_dir / 'audio_visualizer', '--output_format=ppm', *flags16])
    png16 = b''.join((image_dir16 / f'{i}.png').read_bytes() for i in range(expected_frames))
    if not stream16.startswith(b'P6\n17 13\n65535\n') or \
            decode_images(stream16, 'ppm') != decode_images(png16, 'png'):
        raise AssertionError('PPM stdout did not preserve 16-bit PNG pixels')

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
