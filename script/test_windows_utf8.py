#!/usr/bin/env python3
"""Check embedded manifests and Unicode paths in a staged Windows distribution."""
import argparse
import ctypes
from ctypes import wintypes
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET


def check_manifest(executable):
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.LoadLibraryExW.argtypes = [wintypes.LPCWSTR, wintypes.HANDLE, wintypes.DWORD]
    kernel.LoadLibraryExW.restype = wintypes.HMODULE
    kernel.FindResourceW.argtypes = [wintypes.HMODULE, ctypes.c_void_p, ctypes.c_void_p]
    kernel.FindResourceW.restype = wintypes.HANDLE
    kernel.SizeofResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    kernel.SizeofResource.restype = wintypes.DWORD
    kernel.LoadResource.argtypes = [wintypes.HMODULE, wintypes.HANDLE]
    kernel.LoadResource.restype = wintypes.HANDLE
    kernel.LockResource.argtypes = [wintypes.HANDLE]
    kernel.LockResource.restype = ctypes.c_void_p
    kernel.FreeLibrary.argtypes = [wintypes.HMODULE]
    module = kernel.LoadLibraryExW(str(executable), None, 2)  # LOAD_LIBRARY_AS_DATAFILE
    if not module:
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        resource = kernel.FindResourceW(module, 1, 24)  # RT_MANIFEST, process manifest
        assert resource, f'Missing process manifest: {executable}'
        size = kernel.SizeofResource(module, resource)
        data = kernel.LockResource(kernel.LoadResource(module, resource))
        assert size and data, f'Cannot read manifest: {executable}'
        root = ET.fromstring(ctypes.string_at(data, size))
        codepage = root.find('.//{http://schemas.microsoft.com/SMI/2019/WindowsSettings}activeCodePage')
        assert codepage is not None and codepage.text == 'UTF-8', executable
    finally:
        kernel.FreeLibrary(module)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--stage-dir', type=Path, required=True)
    parser.add_argument('--ffmpeg', type=Path, required=True)
    args = parser.parse_args()
    assert os.name == 'nt', 'Run this acceptance check on Windows 10 1903+ or Windows 11'
    repo = Path(__file__).resolve().parents[1]
    stage = args.stage_dir.resolve()
    for executable in (stage / 'bin').glob('*.exe'):
        check_manifest(executable)
    assert (stage / 'bin/phase_limiter.exe').is_file()
    # Python launches via CreateProcessW, independently of the parent's ACP.
    with tempfile.TemporaryDirectory(prefix='phaselimiter-utf8-') as temporary:
        root = Path(temporary) / '日本語 空白 🎵'
        shutil.copytree(stage, root / '配布物 🎵')
        distribution = root / '配布物 🎵'
        ffmpeg = root / 'FFmpeg 実行 🎵.exe'
        shutil.copy2(args.ffmpeg, ffmpeg)
        source = root / '入力 音楽 🎵.wav'
        shutil.copy2(repo / 'test_data/test2.wav', source)
        reference = root / '参照 🎵.json'
        shutil.copy2(repo / 'resource/mastering_reference.json', reference)
        scratch = root / '一時 領域 🎵'
        scratch.mkdir()
        env = os.environ.copy()
        env.update(TEMP=str(scratch), TMP=str(scratch))
        cache = distribution / 'resource/sound_quality2_cache'
        assert cache.is_file(), cache
        for output_format in ('wav', 'mp3', 'aac'):
            output = root / f'出力 音楽 🎵.{output_format}'
            command = [str(distribution / 'bin/phase_limiter.exe'),
                       f'--input={source}', f'--output={output}',
                       f'--output_format={output_format}', f'--ffmpeg={ffmpeg}',
                       f'--tmp={scratch}', '--start_at=0', '--end_at=1',
                       '--mastering', '--mastering_mode=mastering5',
                       '--mastering_reverb=false', f'--sound_quality2_cache={cache}',
                       f'--mastering_reference_file={reference}',
                       '--mastering5_optimization_max_eval_count=2',
                       '--reference_mode=peak', '--reference=-6',
                       '--max_iter1=2', '--max_iter2=8', '--worker_count=1']
            result = subprocess.run(command, env=env, capture_output=True, timeout=300)
            assert result.returncode == 0, result.stderr.decode('utf-8', errors='replace')
            assert output.stat().st_size > 0, output
            subprocess.run([str(ffmpeg), '-v', 'error', '-i', str(output),
                            '-f', 'null', '-'], check=True, env=env, timeout=60)
    print('Embedded UTF-8 manifests and Unicode distribution/input/output/cache/temp/FFmpeg paths passed')


if __name__ == '__main__':
    main()
