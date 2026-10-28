#!/usr/bin/env python3
"""Fetch pinned dependencies, build isolated executables, and collect results."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import uuid

import fixtures

ROOT = Path(__file__).resolve().parent
DEPENDENCIES = json.loads((ROOT / 'dependencies.json').read_text())
BACKENDS = ('accelerate', 'ipp', 'fftw', 'pocketfft', 'kissfft')


def download(name, work):
    spec = DEPENDENCIES[name]
    cache = work / 'downloads' / spec['sha256']
    cache.parent.mkdir(parents=True, exist_ok=True)
    if cache.exists() and fixtures.sha256(cache) == spec['sha256']:
        return cache
    partial = cache.with_suffix('.partial')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(spec['url'], timeout=120) as source, partial.open('wb') as output:
                shutil.copyfileobj(source, output)
            if fixtures.sha256(partial) != spec['sha256']:
                raise ValueError(f'{name}: SHA-256 mismatch')
            partial.replace(cache)
            return cache
        except Exception:
            partial.unlink(missing_ok=True)
            if attempt == 2:
                raise
            time.sleep(2)


def prepare_source(name, work):
    destination = work / 'deps' / name
    stamp = destination / '.fft-experiment-sha256'
    if stamp.exists() and stamp.read_text().strip() == DEPENDENCIES[name]['sha256']:
        return destination
    archive = download(name, work)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temporary:
        temporary = Path(temporary)
        with tarfile.open(archive, 'r:gz') as source:
            # Archives are hash-pinned. Also reject paths/links escaping extraction.
            for member in source.getmembers():
                paths = [temporary / member.name]
                if member.issym():
                    paths.append(temporary / Path(member.name).parent / member.linkname)
                if member.islnk():
                    paths.append(temporary / member.linkname)
                if member.isdev() or any(not path.resolve().is_relative_to(temporary) for path in paths):
                    raise ValueError(f'Unsafe dependency archive member: {member.name}')
            source.extractall(temporary)
        roots = list(temporary.iterdir())
        if len(roots) != 1 or not roots[0].is_dir():
            raise ValueError(f'{name}: expected a single archive root')
        if destination.exists():
            shutil.rmtree(destination)
        shutil.move(str(roots[0]), destination)
    stamp.write_text(DEPENDENCIES[name]['sha256'] + '\n')
    return destination


def prepare_ipp(work):
    if platform.system() != 'Linux' or platform.machine() not in ('x86_64', 'AMD64'):
        raise ValueError('Automatic IPP extraction supports Linux x86_64 only; provide --ipp-root otherwise')
    destination = work / 'deps' / 'ipp'
    signature = '\n'.join(DEPENDENCIES[name]['sha256'] for name in ('ipp-linux-devel', 'ipp-linux-common'))
    stamp = destination / '.fft-experiment-sha256'
    if not stamp.exists() or stamp.read_text() != signature:
        destination.mkdir(parents=True, exist_ok=True)
        for name in ('ipp-linux-devel', 'ipp-linux-common'):
            subprocess.run(['dpkg-deb', '--extract', str(download(name, work)), str(destination)], check=True)
        stamp.write_text(signature)
    return destination / 'opt/intel/oneapi/ipp/2021.9.0'


def checked(command, log):
    print('+', ' '.join(map(str, command)), flush=True)
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open('w') as output:
        result = subprocess.run(list(map(str, command)), stdout=output, stderr=subprocess.STDOUT)
    if result.returncode:
        print('\n'.join(log.read_text(errors='replace').splitlines()[-60:]), flush=True)
        raise RuntimeError(f'Command failed ({result.returncode}); see {log}')


def cpu_model():
    if platform.system() == 'Darwin':
        return subprocess.check_output(['sysctl', '-n', 'machdep.cpu.brand_string'], text=True).strip()
    if Path('/proc/cpuinfo').exists():
        for line in Path('/proc/cpuinfo').read_text().splitlines():
            if line.startswith('model name'):
                return line.split(':', 1)[1].strip()
    return platform.processor()


def default_jobs():
    # Match nproc on Linux, including its affinity/environment restrictions.
    if shutil.which('nproc'):
        return int(subprocess.check_output(['nproc'], text=True))
    return os.cpu_count() or 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--backends', default='pocketfft,kissfft,fftw')
    parser.add_argument('--precisions', default='float,double')
    parser.add_argument('--sizes', default=','.join(map(str, fixtures.DEFAULT_SIZES)))
    parser.add_argument('--work-dir', type=Path, default=ROOT / '.work')
    parser.add_argument('--output-dir', type=Path)
    parser.add_argument('--ipp-root', type=Path)
    parser.add_argument('--jobs', type=int, default=default_jobs())
    parser.add_argument('--timing-ms', type=float, default=10)
    parser.add_argument('--timing-sizes', default='256,1000,1024,4096,4097,8192,16384',
                        help='Comma-separated sizes to time, or all')
    parser.add_argument('--trials', type=int, default=7)
    parser.add_argument('--fftw-planner', choices=('estimate', 'measure'))
    parser.add_argument('--qemu', action='store_true', help='Cross-build Linux AArch64; correctness only')
    args = parser.parse_args()
    args.fftw_planner = args.fftw_planner or ('estimate' if args.qemu else 'measure')
    backends = args.backends.split(',')
    precisions = args.precisions.split(',')
    sizes = tuple(dict.fromkeys(map(int, args.sizes.split(','))))
    if not backends or any(name not in BACKENDS for name in backends):
        parser.error('Unknown backend')
    if any(name not in ('float', 'double') for name in precisions) or not precisions:
        parser.error('Unknown precision')
    if not sizes or min(sizes) < 1 or max(sizes) > 1 << 20 or args.jobs < 1:
        parser.error('Invalid sizes/jobs')
    if not math.isfinite(args.timing_ms) or args.timing_ms < 0 or not 1 <= args.trials <= 100:
        parser.error('Invalid timing duration/trial count')
    timing_sizes = ','.join(map(str, sizes)) if args.timing_sizes == 'all' else args.timing_sizes
    try:
        if any(int(n) < 1 for n in timing_sizes.split(',')):
            raise ValueError()
    except ValueError:
        parser.error('Invalid timing sizes')
    if args.qemu and any(name in ('ipp', 'accelerate') for name in backends):
        parser.error('QEMU supports the portable libraries only')
    work = args.work_dir.resolve()
    work.mkdir(parents=True, exist_ok=True)
    destination = (args.output_dir or work / 'results').resolve()
    destination.mkdir(parents=True, exist_ok=True)
    fixture_dir = work / 'fixtures'
    reference_file = fixture_dir / 'reference.json'
    reference = json.loads(reference_file.read_text()) if reference_file.exists() else {}
    generator_hash = fixtures.sha256(ROOT / 'fixtures.py')
    if reference.get('sizes') != list(sizes) or reference.get('generator_sha256') != generator_hash or any(
            not (fixture_dir / name).exists() or fixtures.sha256(fixture_dir / name) != checksum
            for name, checksum in reference.get('files', {}).items()):
        print('Generating independent 80-digit references...', flush=True)
        reference = fixtures.generate(fixture_dir, sizes)
        reference['generator_sha256'] = generator_hash
        reference_file.write_text(json.dumps(reference, indent=2) + '\n')
    reference_hash = fixtures.sha256(reference_file)
    shutil.copyfile(reference_file, destination / 'reference.json')
    shutil.copyfile(ROOT / 'dependencies.json', destination / 'dependencies.json')
    run_id = str(uuid.uuid4())
    failures = []
    pending = []
    for backend in backends:
        if backend in ('pocketfft', 'kissfft', 'fftw'):
            prepare_source(backend, work)
        ipp_root = (args.ipp_root.resolve() if args.ipp_root else prepare_ipp(work)) if backend == 'ipp' else None
        for precision in precisions:
            label = f'{backend}-{precision}'
            build = work / 'build' / ('qemu' if args.qemu else 'native') / label
            result_dir = destination / label
            result_dir.mkdir(parents=True, exist_ok=True)
            for name in ('results.jsonl', 'outputs.bin', 'execution.log'):
                (result_dir / name).unlink(missing_ok=True)
            source_files = {str(path.relative_to(ROOT)): fixtures.sha256(path) for path in (
                ROOT / 'CMakeLists.txt', ROOT / 'main.cpp', ROOT / 'plan.h', ROOT / 'fixtures.py',
                ROOT / 'run.py', ROOT / 'adapters' / f'{backend}.cpp')}
            metadata = dict(schema=1, backend=backend, precision=precision, run_id=run_id,
                            utc=datetime.now(timezone.utc).isoformat(), platform=platform.platform(),
                            system=platform.system(), host_architecture=platform.machine(), cpu=cpu_model(),
                            target_architecture='aarch64' if args.qemu else platform.machine(),
                            execution='qemu' if args.qemu else 'native', reference_sha256=reference_hash,
                            source_files=source_files,
                            fftw_planner=args.fftw_planner, timing_enabled=not args.qemu and backend != 'ipp'
                            and args.timing_ms > 0)
            (result_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
            configure = ['cmake', '-S', ROOT, '-B', build, '-G', 'Ninja', '-DCMAKE_BUILD_TYPE=Release',
                         f'-DFFT_BACKEND={backend}', f'-DFFT_PRECISION={precision}',
                         f'-DFFT_DEPS_DIR={work / "deps"}', f'-DFFT_BUILD_JOBS={args.jobs}',
                         f'-DFFT_FFTW_PLANNER={args.fftw_planner}']
            if ipp_root:
                configure.append(f'-DFFT_IPP_ROOT={ipp_root}')
            if args.qemu:
                configure.append(f'-DCMAKE_TOOLCHAIN_FILE={ROOT / "toolchains/linux-aarch64.cmake"}')
            try:
                checked(configure, result_dir / 'configure.log')
                checked(['cmake', '--build', build, '--parallel', args.jobs], result_dir / 'build.log')
                command = [build / 'fft_experiment', '--fixtures', fixture_dir, '--output', result_dir,
                           '--timing-ms', 0 if args.qemu or backend == 'ipp' else args.timing_ms,
                           '--trials', args.trials, '--timing-sizes', timing_sizes]
                if args.qemu:
                    command = ['qemu-aarch64', '-L', '/usr/aarch64-linux-gnu', *command]
                pending.append((label, command, result_dir, metadata))
            except RuntimeError as error:
                failures.append(label)
                metadata['error'] = str(error)
            metadata['success'] = False
            (result_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    # Finish all compilation before measuring: compilation load/heat should not
    # be interleaved with one library's timings. No concurrent timed processes.
    for label, command, result_dir, metadata in pending:
        try:
            checked(command, result_dir / 'execution.log')
        except RuntimeError as error:
            failures.append(label)
            metadata['error'] = str(error)
        metadata['success'] = label not in failures
        (result_dir / 'run.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'Results: {destination}', flush=True)
    return bool(failures)


if __name__ == '__main__':
    raise SystemExit(main())
