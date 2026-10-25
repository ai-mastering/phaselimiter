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


def cpu_model():
    if sys.platform == 'darwin':
        try:
            return subprocess.check_output(['/usr/sbin/sysctl', '-n', 'machdep.cpu.brand_string'],
                                           text=True).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if sys.platform.startswith('linux'):
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


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
    units = re.findall(r'unit_eval:([^\s]+)', log)
    if units:
        result['unit_eval'] = float(units[-1])
        result['raw_eval_estimate'] = result['normalized_eval'] * result['unit_eval']
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


def normalized_error_tolerance(reference, actual):
    # Only the Linux x86 -> native macOS ARM comparison gets the wider bound.
    if (reference.get('platform', '').startswith('Linux-')
            and reference.get('architecture') in ('x86_64', 'AMD64')
            and actual.get('platform', '').startswith('macOS-')
            and actual.get('architecture') in ('arm64', 'aarch64')):
        return .10
    return .05


def repeatability_errors(name, fingerprints):
    if len(fingerprints) < 2:
        raise ValueError('Repeatability requires at least two executions')
    first = fingerprints[0]
    errors = []
    for repeat, fingerprint in enumerate(fingerprints[1:], 1):
        changed = sorted(key for key in first.keys() | fingerprint.keys()
                         if first.get(key) != fingerprint.get(key))
        if changed:
            errors.append(f'{name}: repeat {repeat} differs from warmup: {", ".join(changed)}')
    return errors


def compare(reference, actual, reference_dir, actual_dir):
    errors = list(actual.get('repeatability_errors', []))
    errors += ['Linux reference: ' + error for error in reference.get('repeatability_errors', [])]
    rows = []
    relative_tolerance = normalized_error_tolerance(reference, actual)
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
            allowed = max(1e-4, relative_tolerance * abs(before['normalized_eval']))
            if after['normalized_eval'] > before['normalized_eval'] + allowed:
                errors.append(f'{name}: final normalized limiter error regressed by more than '
                              f'{relative_tolerance:.0%} ({before["normalized_eval"]:.6g} -> '
                              f'{after["normalized_eval"]:.6g}; allowed increase {allowed:.6g})')
        rows.append(row)
    return dict(errors=errors, cases=rows, normalized_error_relative_tolerance=relative_tolerance)


def run_cases(bin_dir, cache, output, runs):
    env = dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1', VECLIB_MAXIMUM_THREADS='1')
    inputs = {'test2': REPO / 'test_data/test2.wav', 'short': REPO / 'test_data/test5.wav'}
    report = dict(schema=2, platform=platform.platform(), architecture=platform.machine(), cpu_model=cpu_model(),
                  cache_sha256=digest(cache), input_sha256={k: digest(v) for k, v in inputs.items()},
                  measured_runs=runs, warmup_runs=1, repeatability_errors=[], cases={})
    analyzer = bin_dir / 'audio_analyzer'
    limiter = bin_dir / 'phase_limiter'
    report['binary_sha256'] = {name: digest(bin_dir / name) for name in ('audio_analyzer', 'phase_limiter')}
    common = ['--worker_count=1', f'--sound_quality2_cache={cache}']

    def execute(name, command, repetitions, fingerprint):
        times, fingerprints = [], []
        for repeat in range(repetitions + 1):  # One unmeasured warmup.
            started = time.perf_counter()
            result = subprocess.run([str(x) for x in command], cwd=output, env=env,
                                    capture_output=True, text=True, timeout=300)
            elapsed = time.perf_counter() - started
            (output / f'{name}-{repeat}.stdout').write_text(result.stdout)
            (output / f'{name}-{repeat}.stderr').write_text(result.stderr)
            if result.returncode:
                raise RuntimeError(f'{name} failed ({result.returncode}):\n{result.stderr[-8000:]}')
            fingerprints.append(fingerprint(result))
            if repeat:
                times.append(elapsed)
        report['repeatability_errors'].extend(repeatability_errors(name, fingerprints))
        (output / f'{name}-repeats.json').write_text(json.dumps(fingerprints, indent=2, allow_nan=False))
        return times, result, fingerprints

    def analyze(name, source, repetitions):
        def fingerprint(result):
            data = json.loads(result.stdout)
            require_finite(data)
            if not isinstance(data.get('sound_quality2'), (int, float)):
                raise ValueError(f'{name}: model-cache score is missing')
            if not 0 <= data['sound_quality2'] <= 1:
                raise ValueError(f'{name}: model-cache score is out of range')
            canonical = json.dumps(data, sort_keys=True, separators=(',', ':'), allow_nan=False)
            return dict(analysis_sha256=hashlib.sha256(canonical.encode()).hexdigest())
        seconds, result, fingerprints = execute(name, [analyzer, f'--input={source}', *common],
                                                repetitions, fingerprint)
        return dict(seconds=seconds, analysis=json.loads(result.stdout), repeats=fingerprints)

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
        def fingerprint(result):
            samples = decode_audio(audio)
            return dict(pcm_sha256=hashlib.sha256(samples.tobytes()).hexdigest(),
                        **convergence(result.stderr))
        seconds, result, fingerprints = execute(name, command, runs, fingerprint)
        record = analyze(name + '-output', audio, 1)
        record['output_analysis_repeats'] = record.pop('repeats')
        record.update(seconds=seconds, audio=audio.name, convergence=convergence(result.stderr),
                      repeats=fingerprints)
        report['cases'][name] = record
    return report


def summary(report, comparison):
    lines = ['## Native DSP validation', '',
             f'Environment: {report["platform"]} ({report["architecture"]})',
             f'CPU: {report.get("cpu_model", "unknown")}', '',
             'One worker, identical inputs and Linux-generated cache; no QEMU.', '',
             'Same-environment repeatability: ' + ('FAIL' if report.get('repeatability_errors') else 'PASS')
             + ' (warmup and measured runs; exact analysis, PCM samples, and reported convergence).', '']
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
                  'limiter waveform relative RMS error at most 2%, final normalized error at most '
                  f'{comparison["normalized_error_relative_tolerance"]:.0%} worse.', '',
                  '**Result: ' + ('FAIL' if comparison['errors'] else 'PASS') + '**']
        lines += ['- ' + error for error in comparison['errors']]
    else:
        lines += ['Linux reference recorded. Native macOS jobs compare against this artifact.']
        lines += ['- ' + error for error in report.get('repeatability_errors', [])]
        if report.get('repeatability_errors'):
            lines += ['', '**Result: FAIL**']
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
    return int(bool(report['repeatability_errors'] or (comparison and comparison['errors'])))


if __name__ == '__main__':
    sys.exit(main())
