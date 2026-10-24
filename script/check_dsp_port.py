#!/usr/bin/env python3
"""Record native DSP results and compare them with the Linux CI reference."""
import argparse
from array import array
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import re
import statistics
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]
# Same input/cache and one worker isolate DSP differences from parallel ordering.
DB_FIELDS = ('peak', 'rms', 'true_peak', 'lowpass_true_peak_15khz', 'loudness',
             'youtube_loudness', 'loudness_range', 'loudness_range_short', 'drr')
EXACT_FIELDS = ('frames', 'sample_rate', 'channels')


def digest(path):
    checksum = hashlib.sha256()
    with path.open('rb') as source:
        for block in iter(lambda: source.read(1024 * 1024), b''):
            checksum.update(block)
    return checksum.hexdigest()


def require_finite(value, path=''):
    if isinstance(value, dict):
        for key, item in value.items():
            require_finite(item, path + '/' + key)
    elif isinstance(value, list):
        for index, item in enumerate(value):
            require_finite(item, path + '/' + str(index))
    elif isinstance(value, (int, float)) and not math.isfinite(value):
        raise ValueError(f'Non-finite value at {path}: {value}')


def compare_analysis(reference, actual, db_tolerance, score_tolerance):
    require_finite(reference)
    require_finite(actual)
    errors = []
    for key in EXACT_FIELDS:
        if reference[key] != actual[key]:
            errors.append(f'{key}: {reference[key]} -> {actual[key]}')
    deltas = {}
    for key in (*DB_FIELDS, 'sound_quality2'):
        delta = abs(actual[key] - reference[key])
        deltas[key] = delta
        tolerance = score_tolerance if key == 'sound_quality2' else db_tolerance
        if delta > tolerance:
            errors.append(f'{key}: delta {delta:.6g} exceeds {tolerance:g}')
    return deltas, errors


def convergence(log):
    final = re.findall(r'normalized_eval:([^\s]+)\s+last_iter:(\d+)\s+last_iter2:(\d+)', log)
    correct = re.findall(r'normalized_eval:([^\s]+)\s+limiting_error:([^\s]+)', log)
    if not final or not correct:
        raise ValueError('Missing final limiter convergence diagnostics')
    result = dict(normalized_eval=float(correct[-1][0]), limiting_error=float(correct[-1][1]),
                  iterations1=int(final[-1][1]), iterations2=int(final[-1][2]))
    require_finite(result)
    if result['normalized_eval'] < 0 or result['limiting_error'] < 0:
        raise ValueError('Negative limiter error')
    return result


def decode_audio(path):
    result = subprocess.run(['ffmpeg', '-v', 'error', '-i', str(path), '-f', 'f32le',
                             '-acodec', 'pcm_f32le', '-'], capture_output=True, check=True)
    samples = array('f')
    samples.frombytes(result.stdout)
    if sys.byteorder != 'little':
        samples.byteswap()
    if not samples or any(not math.isfinite(x) for x in samples):
        raise ValueError(f'Empty or non-finite audio: {path}')
    return samples


def compare_samples(reference, actual):
    if not reference or len(reference) != len(actual):
        raise ValueError(f'Audio sample count differs: {len(reference)} != {len(actual)}')
    if any(not math.isfinite(x) for samples in (reference, actual) for x in samples):
        raise ValueError('Non-finite audio samples')
    energy = math.fsum(float(x) * x for x in reference)
    error = math.fsum((float(x) - y) ** 2 for x, y in zip(reference, actual))
    relative = math.sqrt(error / max(energy, 1e-30))
    return dict(relative_rms_error=relative,
                max_abs_error=max(abs(float(x) - y) for x, y in zip(reference, actual)),
                snr_db=None if error == 0 else -20 * math.log10(max(relative, 1e-30)))


def compare(reference, actual, reference_dir, actual_dir):
    errors, rows = [], []
    for key in ('schema', 'cache_sha256', 'input_sha256'):
        if reference[key] != actual[key]:
            raise ValueError(f'Reference {key} differs; regenerate the Linux reference')
    if reference['cases'].keys() != actual['cases'].keys():
        raise ValueError('Reference case set differs')
    for name, record in actual['cases'].items():
        baseline = reference['cases'][name]
        is_input = name.startswith('analyzer-')
        # Final audio can differ slightly after iterative optimization. The
        # original-input analysis uses tighter bounds, including the LOF score.
        deltas, failures = compare_analysis(baseline['analysis'], record['analysis'],
                                           .1 if is_input else .25,
                                           .005 if is_input else .02)
        errors.extend(name + ': ' + failure for failure in failures)
        row = dict(case=name, analysis_deltas=deltas,
                   linux_seconds=statistics.median(baseline['seconds']),
                   native_seconds=statistics.median(record['seconds']))
        if 'audio' in record:
            row['audio'] = compare_samples(decode_audio(reference_dir / baseline['audio']),
                                           decode_audio(actual_dir / record['audio']))
            # A deterministic limiter should keep the waveform close. Mastering
            # additionally optimizes parameters, so its waveform difference is
            # reported while its objective and output metrics are checked.
            if name == 'limiter' and row['audio']['relative_rms_error'] > .02:
                errors.append(name + ': relative waveform RMS error exceeds 2%')
            before, after = baseline['convergence'], record['convergence']
            row['convergence'] = dict(linux=before, native=after)
            allowed = max(1e-4, .05 * abs(before['normalized_eval']))
            if after['normalized_eval'] > before['normalized_eval'] + allowed:
                errors.append(name + ': final normalized limiter error regressed by more than 5%')
        rows.append(row)
    return dict(errors=errors, cases=rows)


def run_cases(bin_dir, cache, output, runs):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    inputs = {'test2': REPO / 'test_data/test2.wav', 'short': REPO / 'test_data/test5.wav'}
    report = dict(schema=1, platform=platform.platform(), architecture=platform.machine(),
                  cache_sha256=digest(cache), input_sha256={k: digest(v) for k, v in inputs.items()}, cases={})
    analyzer = bin_dir / 'audio_analyzer'
    limiter = bin_dir / 'phase_limiter'
    common = ['--worker_count=1', f'--sound_quality2_cache={cache}']

    def execute(name, command, repetitions):
        times = []
        for repeat in range(repetitions + 1):  # One unmeasured warmup.
            started = time.perf_counter()
            result = subprocess.run([str(x) for x in command], cwd=output, env=env,
                                    capture_output=True, text=True, timeout=300)
            elapsed = time.perf_counter() - started
            (output / f'{name}-{repeat}.stdout').write_text(result.stdout)
            (output / f'{name}-{repeat}.stderr').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'{name} failed ({result.returncode}):\n{result.stderr[-8000:]}')
            if repeat:
                times.append(elapsed)
        return times, result

    def analyze(name, source, repetitions):
        seconds, result = execute(name, [analyzer, f'--input={source}', *common], repetitions)
        data = json.loads(result.stdout)
        require_finite(data)
        if not isinstance(data.get('sound_quality2'), (int, float)):
            raise ValueError(f'{name}: model-cache score is missing')
        if not 0 <= data['sound_quality2'] <= 1:
            raise ValueError(f'{name}: model-cache score is out of range')
        return dict(seconds=seconds, analysis=data)

    for name, source in inputs.items():
        report['cases']['analyzer-' + name] = analyze('analyzer-' + name, source, runs)
    for name in ('limiter', 'mastering5'):
        audio = output / (name + '.wav')
        command = [limiter, f'--input={inputs["test2"]}', f'--output={audio}',
                   f'--tmp={output / "tmp"}', '--bit_depth=32', *common,
                   f'--mastering_reference_file={REPO / "resource/mastering_reference.json"}']
        if name == 'mastering5':
            command += ['--mastering=true', '--mastering_mode=mastering5',
                        '--mastering5_optimization_algorithm=nm',
                        '--mastering5_optimization_max_eval_count=4000',
                        '--mastering5_optimization_seed=1',
                        f'--mastering5_mastering_reference_file={REPO / "resource/mastering_reference.json"}']
        seconds, result = execute(name, command, runs)
        record = analyze(name + '-output', audio, 1)
        decode_audio(audio)  # Fail on non-finite or empty output even on Linux.
        record.update(seconds=seconds, audio=audio.name, convergence=convergence(result.stderr))
        report['cases'][name] = record
    return report


def summary(report, comparison):
    lines = ['## Native DSP validation', '',
             f'Environment: {report["platform"]} ({report["architecture"]})', '',
             'One worker, identical inputs and Linux-generated cache; no QEMU.', '']
    if comparison:
        lines += ['| Case | Linux seconds | Native seconds | Audio SNR dB |',
                  '|---|---:|---:|---:|']
        for row in comparison['cases']:
            audio = row.get('audio')
            snr = '-' if not audio else ('identical' if audio['snr_db'] is None else f'{audio["snr_db"]:.2f}')
            lines.append(f'| {row["case"]} | {row["linux_seconds"]:.3f} | {row["native_seconds"]:.3f} | {snr} |')
        lines += ['', 'Timing medians include process/FFmpeg startup. Runner hardware may differ;',
                  'times are recorded for inspection and are not a speed regression gate.', '',
                  'Checks: original-input metrics within 0.1 dB, cache score within 0.005;',
                  'final-output metrics within 0.25 dB and score within 0.02;',
                  'limiter waveform relative RMS error at most 2%, final normalized error at most 5% worse.', '',
                  '**Result: ' + ('FAIL' if comparison['errors'] else 'PASS') + '**']
        lines += ['- ' + error for error in comparison['errors']]
    else:
        lines += ['Linux reference recorded. Native macOS jobs compare against this artifact.']
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bin-dir', type=Path, required=True)
    parser.add_argument('--sound-quality-cache', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--runs', type=int, default=3)
    args = parser.parse_args()
    if args.runs < 1:
        parser.error('--runs must be positive')
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report = run_cases(args.bin_dir.resolve(), args.sound_quality_cache.resolve(), output, args.runs)
    (output / 'results.json').write_text(json.dumps(report, indent=2, allow_nan=False))
    comparison = None
    if args.reference:
        reference = args.reference.resolve()
        comparison = compare(json.loads((reference / 'results.json').read_text()), report, reference, output)
        (output / 'comparison.json').write_text(json.dumps(comparison, indent=2, allow_nan=False))
    text = summary(report, comparison)
    (output / 'summary.md').write_text(text)
    print(text)
    if os.environ.get('GITHUB_STEP_SUMMARY'):
        with open(os.environ['GITHUB_STEP_SUMMARY'], 'a') as target:
            target.write(text)
    return int(bool(comparison and comparison['errors']))


if __name__ == '__main__':
    sys.exit(main())
