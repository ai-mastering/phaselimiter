#!/usr/bin/env python3
"""Run CLI integration, regression, benchmark, or valgrind checks outside the source tree."""
import argparse
import itertools
from pathlib import Path
import shlex
import subprocess
import sys


def run_checks(mode, bin_dir, output, cache):
    repo = Path(__file__).resolve().parents[1]
    output.mkdir(parents=True, exist_ok=True)
    # Keep temporary encodes on the output filesystem so the final rename works
    # when /workspace is a Docker bind mount and /tmp is container-local.
    limiter_command = [str(bin_dir / 'phase_limiter'), f'--tmp={output / "tmp"}']
    analyzer = str(bin_dir / 'audio_analyzer')
    wave = str(repo / 'test_data/test2.wav')
    short_wave = str(repo / 'test_data/test5.wav')
    reference = str(repo / 'resource/mastering_reference.json')

    def run(name, command, instrument=True):
        command = [str(value) for value in command]
        if mode == 'valgrind' and instrument:
            command = ['valgrind', '--error-exitcode=1',
                       '--leak-check=full', '--show-leak-kinds=all',
                       '--errors-for-leak-kinds=definite,indirect',
                       '--track-origins=yes',
                       f'--log-file={output / (name + ".valgrind.log")}', *command]
        print(shlex.join(command), flush=True)
        log_path = output / (name + '.log')
        with log_path.open('w') as log:
            result = subprocess.run(command, cwd=repo if name == 'unit' else output,
                                    stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            print(f'Check {name} failed; output from {log_path}:', flush=True)
            print(log_path.read_text(errors='replace'), flush=True)
            if mode == 'valgrind' and instrument:
                print((output / (name + '.valgrind.log')).read_text(errors='replace'), flush=True)
            result.check_returncode()

    def limit(name, *flags, input_file=wave):
        run(name, [*limiter_command, f'--input={input_file}', f'--output={output / (name + ".wav")}',
                   f'--mastering_reference_file={reference}', *flags])

    def analyze():
        run('audio_analyzer', [analyzer, f'--input={short_wave}',
            f'--analysis_data_dir={repo / "resource/analysis_data"}', f'--sound_quality2_cache={cache}',
            f'--spectrogram_output={output / "spectrogram.png"}',
            f'--spectrum_distribution_output={output / "spectrum_distribution.png"}',
            f'--stereo_distribution_output={output / "stereo_distribution.png"}'])

    if mode == 'integration':
        analyze()
        limit('ceiling', '--ceiling=-24', '--ceiling_mode=true_peak',
              '--reference_mode=youtube_loudness', '--reference=-10.3')
        run('mp3', [*limiter_command, f'--input={wave}', f'--output={output / "quiet.mp3"}',
                    '--reference=-12', '--output_format=mp3'])
        for start, end, rate in [(0, 3, 44100), (0, 1000, 44100), (0.7, 1.7, 48000)]:
            limit(f'range_{end}_{rate}', '--mastering', f'--start_at={start}',
                  f'--end_at={end}', f'--sample_rate={rate}')
        run('range_mp3', [*limiter_command, f'--input={wave}', f'--output={output / "range.mp3"}',
            f'--mastering_reference_file={reference}', '--mastering', '--start_at=0.7',
            '--end_at=1.7', '--output_format=mp3'])
        run('mastering_mp3', [*limiter_command, f'--input={wave}', f'--output={output / "mastering.mp3"}',
            f'--mastering_reference_file={reference}', '--mastering', '--mastering_reverb=false',
            '--output_format=mp3', '--limiter_external_oversample=2',
            f'--limiting_error_spectrogram_output={output / "limiting_error.png"}',
            f'--grad_output={output / "grad.wav"}'])
        limit('mastering3', '--mastering', '--mastering_mode=mastering3', '--mastering3_iteration=10')
        limit('mastering5', '--mastering', '--mastering_mode=mastering5', f'--sound_quality2_cache={cache}')
    elif mode == 'regression':
        # Preserve the historic FMA/segmentation-fault parameter sweep, with
        # checked exit codes instead of the old Ruby script's warn-and-continue.
        iterations = [25, 50, 75, 100, 150, 200, 300, 400, 800, 1600, 3200, 6400, 12800, 25600]
        for loudness, count in itertools.product([-7, -5, -3], iterations):
            limit(f'noise_{loudness}_{count}', '--erb_eval_func_weighting=true',
                  f'--reference={loudness}', f'--max_iter1={count}',
                  f'--max_iter2={10 * count}', '--noise_update_initial_noise=1')
    elif mode == 'benchmark':
        run('google_benchmark', [bin_dir / 'bench', '--benchmark_min_time=0.01',
                                f'--benchmark_out={output / "google_benchmark.json"}',
                                '--benchmark_out_format=json'])
        commands = [
            [analyzer, f'--input={wave}', f'--sound_quality2_cache={cache}'],
            [*limiter_command, f'--input={wave}', f'--output={output / "benchmark.wav"}', f'--mastering_reference_file={reference}'],
            [*limiter_command, f'--input={wave}', f'--output={output / "benchmark_mastering.wav"}',
             '--mastering=true', '--mastering_mode=mastering5', f'--sound_quality2_cache={cache}',
             f'--mastering_reference_file={reference}']]
        run('hyperfine', ['hyperfine', '--warmup', '1', '--runs', '10',
                         '--export-json', output / 'hyperfine.json',
                         *[shlex.join(command) for command in commands]])
    else:
        analyze()
        common = ['--mastering=true', '--mastering_reverb=false', '--max_iter1=20']
        limit('limiter', *common, '--limiter_external_oversample=2', input_file=short_wave)
        limit('mastering3', *common, '--mastering_mode=mastering3', '--mastering3_iteration=10',
              '--reference=-10', '--reference_mode=peak', input_file=short_wave)
        for ceiling in ['peak', 'lowpass_true_peak']:
            limit('mastering5_' + ceiling, *common, '--mastering_mode=mastering5',
                  f'--sound_quality2_cache={cache}', '--reference=-10', '--reference_mode=peak',
                  f'--mastering5_mastering_reference_file={reference}', f'--ceiling_mode={ceiling}',
                  input_file=short_wave)
        run('auxiliary-cli', [sys.executable, repo / 'script/test_auxiliary_cli.py',
            '--bin-dir', bin_dir, '--output-dir', output, '--valgrind'], instrument=False)
        run('unit', [bin_dir / 'phaselimiter_tests',
                    '--gtest_filter=-JsFilterTest*:ClearMixerFilter*:SoundQuality2Calculator.Validation'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=['integration', 'regression', 'benchmark', 'valgrind'], required=True)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--sound-quality-cache', type=Path, required=True)
    args = parser.parse_args()
    run_checks(args.mode, args.bin_dir.resolve(), args.output_dir.resolve(), args.sound_quality_cache.resolve())


if __name__ == '__main__':
    main()
