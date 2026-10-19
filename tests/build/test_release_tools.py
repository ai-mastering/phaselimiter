"""Contracts for SDK integrity and release layout; no native compiler required."""
import hashlib
import importlib.util
from pathlib import Path
import sys
import tarfile
import tempfile
import unittest
from unittest import mock
import zipfile

sys.dont_write_bytecode = True
REPO = Path(__file__).resolve().parents[2]


def load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / 'script' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


fetcher = load('fetch_build_dependencies')
stager = load('stage_release')
packager = load('package_release')


class ReleaseContracts(unittest.TestCase):
    def test_download_rejects_corrupted_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'sdk.tar.bz2'
            source.write_bytes(b'verified SDK')
            cache = root / 'cache'
            cache.mkdir()
            asset = {'url': source.as_uri(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest()}
            downloaded = fetcher.fetch(asset, cache)
            self.assertEqual(downloaded.read_bytes(), b'verified SDK')
            downloaded.write_bytes(b'corrupted SDK')
            with self.assertRaisesRegex(ValueError, 'Checksum mismatch'):
                fetcher.fetch(asset, cache)

    def test_stage_does_not_import_source_cache_and_archives_preserve_layout(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            repo = root / 'source'
            (repo / 'script').mkdir(parents=True)
            (repo / 'packaging/licenses').mkdir(parents=True)
            (repo / 'resource').mkdir()
            (repo / 'packaging/release_files.txt').write_text(
                (REPO / 'packaging/release_files.txt').read_text())
            (repo / 'LICENSE').write_text('project license')
            (repo / 'README.md').write_text('documentation')
            (repo / 'Pipfile').write_text('legacy dependency manifest')
            (repo / 'Pipfile.lock').write_text('legacy lockfile')
            (repo / '.python-version').write_text('3.6')
            (repo / 'script/create_reference.py').write_text('import sklearn')
            (repo / 'test_data').mkdir()
            (repo / 'test_data/input.wav').write_bytes(b'test audio')
            (repo / 'packaging/licenses/notice.txt').write_text('license')
            (repo / 'resource/reference.json').write_text('{}')
            (repo / 'resource/sound_quality2_cache').write_text('stale source cache')
            (repo / 'resource/.DS_Store').write_text('metadata')
            cache = root / 'generated-cache'
            cache.write_text('fresh cache')

            def install(command, check):
                prefix = Path(command[command.index('--prefix') + 1])
                (prefix / 'bin').mkdir(parents=True)
                (prefix / 'bin/phase_limiter').write_text('installed executable')
                (prefix / 'bin/audio_detector').write_text('native detector')
                (prefix / 'lib').mkdir()
                (prefix / 'lib/runtime.so').write_text('installed runtime')

            with mock.patch.object(stager, '__file__', str(repo / 'script/stage_release.py')):
                with mock.patch.object(stager.subprocess, 'run', side_effect=install):
                    prefix = stager.stage(root / 'build', root / 'stage', 'Release', cache)
                    without_cache = stager.stage(root / 'build', root / 'no-cache', 'Release')
                    with self.assertRaisesRegex(ValueError, 'Stage under build'):
                        stager.stage(root / 'build', repo / 'resource/stage', 'Release')
                    with self.assertRaisesRegex(ValueError, 'must be empty'):
                        stager.stage(root / 'build', root / 'stage', 'Release')
            self.assertFalse((without_cache / 'resource/sound_quality2_cache').exists())
            self.assertEqual((prefix / 'resource/sound_quality2_cache').read_text(), 'fresh cache')
            self.assertFalse((prefix / 'resource/.DS_Store').exists())
            self.assertEqual((repo / 'resource/sound_quality2_cache').read_text(), 'stale source cache')
            self.assertEqual((prefix / 'licenses/notice.txt').read_text(), 'license')
            self.assertTrue((prefix / 'lib/runtime.so').exists())
            for entry in ['script', 'test_data', 'Pipfile', 'Pipfile.lock', '.python-version']:
                self.assertFalse((prefix / entry).exists(), entry)
                self.assertTrue((repo / entry).exists(), entry)

            expected = {'phaselimiter/bin/phase_limiter', 'phaselimiter/bin/audio_detector',
                        'phaselimiter/lib/runtime.so',
                        'phaselimiter/resource/reference.json', 'phaselimiter/resource/sound_quality2_cache',
                        'phaselimiter/licenses/notice.txt'}
            with self.assertRaisesRegex(ValueError, 'outside the staging'):
                packager.package(root / 'stage', prefix / 'release.zip')
            for extension in ['.zip', '.tar.xz']:
                archive = root / ('release' + extension)
                packager.package(root / 'stage', archive)
                if extension == '.zip':
                    with zipfile.ZipFile(archive) as package:
                        entries = set(package.namelist())
                else:
                    with tarfile.open(archive) as package:
                        entries = set(package.getnames())
                self.assertTrue(expected <= entries)
                self.assertFalse(any('/source/' in entry or '/build/' in entry for entry in entries))
                self.assertFalse(any('/script/' in entry or '/test_data/' in entry for entry in entries))
                self.assertFalse(any(Path(entry).name in {'Pipfile', 'Pipfile.lock', '.python-version'}
                                     for entry in entries))


if __name__ == '__main__':
    unittest.main()
