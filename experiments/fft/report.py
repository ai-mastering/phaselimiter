#!/usr/bin/env python3
"""Summarize theory errors and compare canonical outputs, never cross-host speed."""
import argparse
import itertools
import json
import math
from pathlib import Path
import struct


def identity(row):
    return row['domain'], row['n'], row['operation'], row['case']


def load(directory):
    metadata = json.loads((directory / 'run.json').read_text())
    rows = []
    results = directory / 'results.jsonl'
    parse_error = None
    if results.exists():
        for line in results.read_text().splitlines():
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as error:
                parse_error = str(error)
                break
    cases = {identity(row): row for row in rows if row['type'] == 'accuracy'}
    skips = {identity(row) for row in rows if row['type'] == 'skip'}
    summaries = [row for row in rows if row['type'] == 'summary']
    label = '/'.join((metadata['system'], metadata['target_architecture'], metadata['execution'],
                      metadata['backend'], metadata['precision']))
    return dict(directory=directory, metadata=metadata, rows=rows, cases=cases, skips=skips,
                summary=summaries[-1] if summaries else None, label=label, parse_error=parse_error)


def read_output(stream, row):
    stream.seek(row['output_offset'] * 16)
    data = stream.read(row['output_count'] * 16)
    if len(data) != row['output_count'] * 16:
        raise ValueError('Truncated output snapshots')
    return [complex(re, im) for re, im in struct.iter_unpack('<dd', data)]


def compare_values(reference, actual):
    if len(reference) != len(actual):
        raise ValueError('Snapshot dimensions differ')
    if any(not math.isfinite(x.real) or not math.isfinite(x.imag) for x in (*reference, *actual)):
        return dict(finite=False, max_abs_difference=None, relative_l2_difference=None, max_difference_index=None)
    errors = [abs(x - y) for x, y in zip(reference, actual)]
    maximum = max(errors, default=0)
    energy = math.fsum(x.real * x.real + x.imag * x.imag for x in reference)
    relative = math.sqrt(math.fsum(x * x for x in errors) / energy) if energy else None
    return dict(finite=True, max_abs_difference=maximum, relative_l2_difference=relative,
                max_difference_index=errors.index(maximum) if errors else 0)


def compare_runs(reference, actual):
    if reference['metadata']['reference_sha256'] != actual['metadata']['reference_sha256']:
        raise ValueError('Reference fixture fingerprints differ; regenerate using the same source and sizes')
    records = []
    union = reference['cases'].keys() | actual['cases'].keys()
    missing = []
    for key in sorted(union):
        if key not in reference['cases'] and key not in reference['skips']:
            missing.append(('reference', key))
        if key not in actual['cases'] and key not in actual['skips']:
            missing.append(('actual', key))
    with (reference['directory'] / 'outputs.bin').open('rb') as left, \
            (actual['directory'] / 'outputs.bin').open('rb') as right:
        for key in sorted(reference['cases'].keys() & actual['cases'].keys()):
            expected = read_output(left, reference['cases'][key])
            observed = read_output(right, actual['cases'][key])
            result = compare_values(expected, observed)
            result.update(domain=key[0], n=key[1], operation=key[2], case=key[3])
            if key[2] == 'forward' and result['finite']:
                result['dc_abs_difference'] = abs(expected[0] - observed[0])
                if key[1] % 2 == 0:
                    result['nyquist_abs_difference'] = abs(expected[key[1] // 2] - observed[key[1] // 2])
            records.append(result)
    return dict(reference=reference['label'], actual=actual['label'], cases=records,
                missing_cases=missing, common_cases=len(records),
                max_abs_difference=max((row['max_abs_difference'] or 0 for row in records), default=0))


def summarize(directories, require_ipp=False, require_accelerate=False):
    runs = [load(directory) for directory in directories]
    problems = []
    for run in runs:
        summary = run['summary']
        if run['parse_error'] or not run['metadata'].get('success') or not summary or summary['failures'] or not summary['completed']:
            problems.append(f'{run["label"]}: build/execution or accuracy sanity check failed')
    for name, required in (('ipp', require_ipp), ('accelerate', require_accelerate)):
        if required:
            for precision in ('float', 'double'):
                if not any(run['metadata']['backend'] == name and run['metadata']['precision'] == precision
                           and run['summary'] and run['summary']['completed'] for run in runs):
                    problems.append(f'Missing {name} {precision} results')
    comparisons = []
    for i, j in itertools.combinations(range(len(runs)), 2):
        a, b = runs[i], runs[j]
        if a['metadata']['precision'] != b['metadata']['precision'] or not a['summary'] or not b['summary']:
            continue
        # IPP compares accuracy across hosts; PocketFFT is the within-run anchor.
        if a['metadata']['backend'] == 'ipp':
            reference, actual = a, b
        elif b['metadata']['backend'] == 'ipp':
            reference, actual = b, a
        elif a['metadata']['run_id'] == b['metadata']['run_id'] and 'pocketfft' in (
                a['metadata']['backend'], b['metadata']['backend']):
            reference, actual = (a, b) if a['metadata']['backend'] == 'pocketfft' else (b, a)
        else:
            continue
        try:
            comparison = compare_runs(reference, actual)
            comparisons.append(comparison)
            if comparison['missing_cases'] or not comparison['common_cases']:
                problems.append(f'{reference["label"]} vs {actual["label"]}: incomplete case coverage')
            if any(not row['finite'] for row in comparison['cases']):
                problems.append(f'{reference["label"]} vs {actual["label"]}: non-finite output')
        except ValueError as error:
            problems.append(str(error))
    # Performance ratios only among libraries executed in the same invocation.
    timings = []
    for run in runs:
        metadata = run['metadata']
        for row in run['rows']:
            if row['type'] == 'timing':
                if metadata['execution'] != 'native' or metadata['backend'] == 'ipp':
                    problems.append('IPP/QEMU timings must not be used for speed comparisons')
                    continue
                timings.append(dict(row, run_id=metadata['run_id'], label=run['label'], backend=metadata['backend'],
                                    precision=metadata['precision']))
    for row in timings:
        peers = [candidate['median_ns'] for candidate in timings
                 if (candidate['run_id'], candidate['precision'], candidate['n'], candidate['domain'], candidate['operation'])
                 == (row['run_id'], row['precision'], row['n'], row['domain'], row['operation'])]
        row['time_over_fastest_same_run'] = row['median_ns'] / min(peers)
    accuracy = []
    for run in runs:
        cases = list(run['cases'].values())
        accuracy.append(dict(label=run['label'], metadata=run['metadata'], summary=run['summary'],
                             backend_metadata=next((row for row in run['rows'] if row['type'] == 'metadata'), None),
                             max_abs_error=max((row['max_abs_error'] for row in cases), default=None),
                             max_relative_l2_error=max((row['relative_l2_error'] for row in cases
                                                        if row['relative_l2_error'] is not None), default=None),
                             worst_cases=sorted(cases, key=lambda row: row['max_abs_error'], reverse=True)[:10]))
    return dict(schema=1, accuracy=accuracy, comparisons=comparisons, timings=timings, problems=problems)


def markdown(result):
    lines = ['# FFT experiment', '',
             'Errors are against 80-digit Decimal references rounded to binary64. '
             'Cross-library differences are diagnostics, not proof that either library is wrong.', '',
             '| Run | Cases | Unsupported | Sanity failures | Max absolute error | Max relative L2 error |',
             '|---|---:|---:|---:|---:|---:|']
    for row in result['accuracy']:
        summary = row['summary'] or {}
        absolute = f'{row["max_abs_error"]:.6g}' if row['max_abs_error'] is not None else '—'
        relative = f'{row["max_relative_l2_error"]:.6g}' if row['max_relative_l2_error'] is not None else '—'
        lines.append(f'| {row["label"]} | {summary.get("completed", 0)} | {summary.get("skipped", 0)} | '
                     f'{summary.get("failures", "incomplete")} | {absolute} | {relative} |')
    lines += ['', 'Absolute maxima combine different lengths/amplitudes; use per-case JSONL and '
              '`summary.json` to interpret them. Sanity bounds are coarse diagnostics, not promised accuracy.', '',
              '## Output differences', '', '| Reference | Actual | Common cases | Max absolute difference |',
              '|---|---|---:|---:|']
    for row in result['comparisons']:
        lines.append(f'| {row["reference"]} | {row["actual"]} | {row["common_cases"]} | {row["max_abs_difference"]:.6g} |')
    lines += ['', '## Warm execution timings', '',
              'Median time includes adapter packing/copies. Plans are reused. Ratios compare only '
              'the same invocation, precision, domain, length and direction. IPP and QEMU are accuracy-only. '
              'Shared CI timings are observational; no performance pass/fail threshold.', '',
              '| Run | Domain | N | Direction | Median µs | Time / fastest in same run |',
              '|---|---|---:|---|---:|---:|']
    for row in sorted(result['timings'], key=lambda row: (row['run_id'], row['domain'], row['n'], row['operation'], row['backend'])):
        lines.append(f'| {row["label"]} | {row["domain"]} | {row["n"]} | {row["operation"]} | '
                     f'{row["median_ns"] / 1000:.4g} | {row["time_over_fastest_same_run"]:.3f} |')
    if result['problems']:
        lines += ['', '## Problems', '', *('- ' + problem for problem in result['problems'])]
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-ipp', action='store_true')
    parser.add_argument('--require-accelerate', action='store_true')
    args = parser.parse_args()
    directories = sorted({path.parent.resolve() for root in args.results for path in root.rglob('run.json')})
    if not directories:
        parser.error('No run.json files found')
    result = summarize(directories, args.require_ipp, args.require_accelerate)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    (args.output / 'summary.md').write_text(markdown(result))
    print(f'Summary: {args.output / "summary.md"}')
    return bool(result['problems'])


if __name__ == '__main__':
    raise SystemExit(main())
