#!/usr/bin/env python3
"""Reproduce the original impulse failure and attribute its numerical error."""
import argparse
import json
from pathlib import Path
import platform
import shutil

import fixtures
import peak_reference
import run


def validation_problems(records, metadata):
    originals = [row for row in records if row['original_case']]
    problems = []
    expected = {(backend, precision, domain, contract) for backend in metadata['backends']
                for precision in ('float', 'double') for domain in ('real', 'complex')
                for contract in metadata['contracts']}
    observed = {(row['backend'], row['precision'], row['domain'], row['fp_contract']) for row in originals}
    if observed != expected or len(originals) != len(expected):
        problems.append('Original-case precision/domain coverage is incomplete or duplicated')
    for backend, precision, domain, contract in expected:
        controls = {row['fault'] for row in records if row['backend'] == backend and row['precision'] == precision
                    and row['domain'] == domain and row['fp_contract'] == contract and row['filter'] == 'endpoints'}
        if not {'drop_dc', 'drop_nyquist', 'mix_dc_nyquist'}.issubset(controls):
            problems.append('Endpoint negative-control coverage is incomplete')
    for row in records:
        if row['scatter_max_error'] != 0 or row.get('perm_comparison_max_error', 0) != 0:
            problems.append('Overlap-add/scatter or Perm/canonical mismatch')
        if row['filter'] == 'endpoints' and row['fault'] != 'none':
            field = 'product_dc_error' if row['fault'] == 'drop_dc' else 'product_nyquist_error'
            if row[field] < 1e-5 or row['wave_max_abs_error'] < 1e-5:
                problems.append('Endpoint negative control was not detected')
    return problems


def summarize(destination, records, metadata):
    originals = [row for row in records if row['original_case']]
    problems = validation_problems(records, metadata)
    result = dict(metadata=metadata, original_cases=originals, cases=records, problems=problems)
    (destination / 'peak-summary.json').write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    lines = ['# True-peak impulse investigation', '',
             'Original assertion: true_peak = -5.1771939979516901e-07 dB. '
             'This is exactly the float dB result for amplitude 0.999999940395355224609375 '
             '(the predecessor of 1.0f). No production tolerance is changed.', '',
             'The FIR scalar formula and overlap-add schedule are frozen reproductions; '
             'production files are not compiled. Real and complex FFT paths use the same input. '
             'The 80-digit kernel reference is computed from each run’s actual binary64 coefficients '
             'and checked against direct DFT at seven bins, including DC/Nyquist.', '',
             '## Original input and schedule', '',
             '| Backend | Precision | FFT path / caller contraction | Peak amplitude | Peak dB | Original assertion passes |',
             '|---|---|---|---:|---:|---|']
    for row in originals:
        lines.append(f'| {row["backend"]} | {row["precision"]} | {row["domain"]}/{row["fp_contract"]} | '
                     f'{row["peak_amplitude"]:.17g} | {row["peak_db"]:.17g} | {row["passes_original_db_tolerance"]} |')
    lines += ['', '## Position sensitivity with the same filter and schedule', '',
              'Every source impulse is zero-inserted at a multiple of four. DC and Nyquist '
              'of its nonzero FFT block are therefore both +4, and the expected peak is always 1.', '',
              '| Backend | Precision | FFT path | Minimum peak | Maximum peak | Positions outside original dB tolerance |',
              '|---|---|---|---:|---:|---|']
    groups = sorted({(row['backend'], row['precision'], row['domain'], row['fp_contract']) for row in originals})
    for backend, precision, domain, contract in groups:
        positions = [row for row in records if (row['backend'], row['precision'], row['domain'], row['fp_contract'])
                     == (backend, precision, domain, contract) and row['filter'] == 'lowpass'
                     and row['kernel_mode'] == 'native_double' and row['layout'] == 'perm'
                     and row['fault'] == 'none' and row['chunk'] == 16384]
        failures = ', '.join(str(row['position']) for row in positions if not row['passes_original_db_tolerance']) or 'none'
        lines.append(f'| {backend} | {precision} | {domain}/{contract} | '
                     f'{min(row["peak_amplitude"] for row in positions):.17g} | '
                     f'{max(row["peak_amplitude"] for row in positions):.17g} | {failures} |')
    lines += ['', '## Additive error contributions at the expected peak', '',
              'Contributions are in linear amplitude. Their sum is the FFT sample error. '
              'Inverse contributions use a direct long-double sum of the actual product spectrum, '
              'with phase factors generated at 80 digits and stored as binary64. '
              'The sum’s reference floor is around binary64 precision, not float precision.', '',
              '| Backend | Precision | FFT path | Kernel FFT / scalar cast | Forward FFT | Multiply | Inverse FFT |',
              '|---|---|---|---:|---:|---:|---:|']
    for row in originals:
        fields = ('kernel_cast_error_at_peak', 'forward_error_at_peak', 'multiply_error_at_peak', 'inverse_error_at_peak')
        lines.append(f'| {row["backend"]} | {row["precision"]} | {row["domain"]}/{row["fp_contract"]} | '
                     + ' | '.join(f'{row[field]:.9g}' for field in fields) + ' |')
    lines += ['', '## Original-case endpoint and waveform errors', '',
              '| Backend | Precision | FFT path | Input DC | Input Nyquist | Product DC | Product Nyquist | Wave max abs |',
              '|---|---|---|---:|---:|---:|---:|---:|']
    for row in originals:
        fields = ('input_dc_error', 'input_nyquist_error', 'product_dc_error', 'product_nyquist_error', 'wave_max_abs_error')
        lines.append(f'| {row["backend"]} | {row["precision"]} | {row["domain"]}/{row["fp_contract"]} | '
                     + ' | '.join(f'{row[field]:.9g}' for field in fields) + ' |')
    lines += ['', '## Structural and negative-control checks', '',
              f'- Cases: {len(records)}',
              f'- Maximum overlap-add vs independent scatter difference: {max((r["scatter_max_error"] for r in records), default=0):.9g}',
              f'- Maximum Perm vs canonical difference: {max((r.get("perm_comparison_max_error", 0) for r in records), default=0):.9g}',
              '- `endpoints` uses padded coefficients [0.75, 0.25], with DC gain 1 and Nyquist gain 0.5.',
              '- Dropping DC or Nyquist must produce detectable product and waveform errors with that filter.',
              '- Treating the packed DC/Nyquist pair as one complex bin is also deliberately tested as an error.',
              '- The low-pass alone is not a strong Nyquist test: its Nyquist response is almost zero.',
              '- Position, chunk size, double/reference80/scalar kernel construction, and all negative controls '
              'are retained in peak-summary.json. Original waveforms are stored as little-endian binary64 complex pairs.']
    if problems:
        lines += ['', '## Problems', '', *('- ' + problem for problem in sorted(set(problems)))]
    (destination / 'peak-summary.md').write_text('\n'.join(lines) + '\n')
    return problems


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends', default='pocketfft,kissfft,fftw')
    parser.add_argument('--work-dir', type=Path, default=run.ROOT / '.work')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ipp-root', type=Path)
    parser.add_argument('--jobs', type=int, default=run.default_jobs())
    parser.add_argument('--qemu', action='store_true')
    parser.add_argument('--contracts', default='off,on', help='Caller FP contraction modes: off,on')
    args = parser.parse_args()
    backends = args.backends.split(',')
    contracts = args.contracts.split(',')
    if any(name not in run.BACKENDS for name in backends) or args.jobs < 1:
        parser.error('Invalid backends/jobs')
    if args.qemu and any(name in ('accelerate', 'ipp') for name in backends):
        parser.error('QEMU supports the portable implementations only')
    if any(contract not in ('off', 'on') for contract in contracts) or len(contracts) != len(set(contracts)):
        parser.error('Invalid or duplicate contraction modes')
    work = args.work_dir.resolve()
    destination = (args.output_dir or work / 'peak-results').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    reference = work / 'peak-reference'
    print('Generating independent true-peak references...', flush=True)
    peak_reference.generate_common(reference)
    metadata = dict(schema=1, backends=backends, contracts=contracts, platform=platform.platform(), cpu=run.cpu_model(),
                    execution='qemu' if args.qemu else 'native',
                    source_files={str(path.relative_to(run.ROOT)): fixtures.sha256(path) for path in (
                        run.ROOT / 'peak_probe.cpp', run.ROOT / 'peak_reference.py',
                        run.ROOT / 'reproduction/dbesi0.cpp', run.ROOT / 'run_peak_probe.py',
                        run.ROOT / 'fixtures.py', run.ROOT / 'CMakeLists.txt', run.ROOT / 'plan.h',
                        run.ROOT / 'dependencies.json',
                        *(run.ROOT / 'adapters' / f'{name}.cpp' for name in backends))}, kernel_references=[])
    records, refined = [], {}
    for backend in backends:
        if backend in ('pocketfft', 'kissfft', 'fftw'):
            run.prepare_source(backend, work)
        ipp_root = (args.ipp_root.resolve() if args.ipp_root else run.prepare_ipp(work)) if backend == 'ipp' else None
        executables = {}
        for precision in ('double', 'float'):
            build = work / 'build' / ('qemu' if args.qemu else 'native') / f'{backend}-{precision}'
            logs = destination / backend / precision
            logs.mkdir(parents=True, exist_ok=True)
            command = ['cmake', '-S', run.ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                       f'-DFFT_BACKEND={backend}', f'-DFFT_PRECISION={precision}',
                       f'-DFFT_DEPS_DIR={work / "deps"}', f'-DFFT_BUILD_JOBS={args.jobs}', '-DFFT_FFTW_PLANNER=estimate']
            if ipp_root:
                command.append(f'-DFFT_IPP_ROOT={ipp_root}')
            if args.qemu:
                command.append(f'-DCMAKE_TOOLCHAIN_FILE={run.ROOT / "toolchains/linux-aarch64.cmake"}')
            run.checked(command, logs / 'configure.log')
            targets = {contract: 'fft_peak_probe' + ('_contract_on' if contract == 'on' else '') for contract in contracts}
            run.checked(['cmake', '--build', build, '--target', *targets.values(), '--parallel', args.jobs], logs / 'build.log')
            for contract, target in targets.items():
                executables[precision, contract] = [build / target]
                if args.qemu:
                    executables[precision, contract] = ['qemu-aarch64', '-L', '/usr/aarch64-linux-gnu', *executables[precision, contract]]
        for contract in contracts:
            for domain in ('real', 'complex'):
                kernels = destination / backend / f'kernels-{domain}-{contract}'
                kernels.mkdir(parents=True, exist_ok=True)
                run.checked([*executables['double', contract], '--output', kernels, '--reference', reference,
                             '--domain', domain, '--prepare', 'true'], kernels / 'prepare.log')
                for filter_name in ('lowpass', 'endpoints'):
                    signature = fixtures.sha256(kernels / f'fir_{filter_name}.bin')
                    if signature not in refined:
                        info = peak_reference.refine_kernel(kernels, filter_name)
                        refined[signature] = (kernels / f'reference_{filter_name}.bin', info)
                    else:
                        source, info = refined[signature]
                        shutil.copyfile(source, kernels / f'reference_{filter_name}.bin')
                    metadata['kernel_references'].append(dict(info, backend=backend, domain=domain, fp_contract=contract))
                for precision in ('double', 'float'):
                    output = destination / backend / precision / f'{domain}-{contract}'
                    output.mkdir(parents=True, exist_ok=True)
                    run.checked([*executables[precision, contract], '--output', output, '--reference', reference,
                                 '--domain', domain, '--kernels', kernels], output / 'execution.log')
                    for line in (output / 'peak.jsonl').read_text().splitlines():
                        row = json.loads(line)
                        if row['type'] == 'case':
                            row.update(backend=backend, precision=precision, domain=domain, fp_contract=contract)
                            records.append(row)
    problems = summarize(destination, records, metadata)
    print(f'Peak investigation: {destination / "peak-summary.md"}', flush=True)
    return bool(problems)


if __name__ == '__main__':
    raise SystemExit(main())
