import copy
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import tempfile
import unittest

import fixtures
import report


class ReferenceTests(unittest.TestCase):
    def test_analytic_forward_matches_independent_direct_dft(self):
        with localcontext() as context:
            context.prec = 80
            for n in (1, 2, 3, 5, 8, 15, 16, 31):
                for real in (False, True):
                    for name in ('zero', 'dc', 'impulse_start', 'impulse_end', 'mixed'):
                        with self.subTest(n=n, real=real, case=name):
                            values, expected = fixtures.analytic_signal(n, real, name)
                            direct = fixtures.direct_dft(values)[:len(expected)]
                            self.assertLess(max(abs(x - y) for a, b in zip(direct, expected)
                                                for x, y in zip(a, b)), D('1e-70'))

    def test_inverse_reference_is_independent_and_handles_odd_real_spectra(self):
        with localcontext() as context:
            context.prec = 80
            for n in (1, 2, 3, 7, 8, 16):
                for real in (False, True):
                    with self.subTest(n=n, real=real):
                        spectrum, expected = fixtures.independent_inverse(n, real)
                        full = list(spectrum)
                        if real:
                            full += [(spectrum[k][0], -spectrum[k][1])
                                     for k in range((n - 1) // 2, 0, -1)]
                        conjugate = [(re, -im) for re, im in full]
                        inverse = [(re / n, -im / n) for re, im in fixtures.direct_dft(conjugate)]
                        self.assertLess(max(abs(x - y) for a, b in zip(inverse, expected)
                                            for x, y in zip(a, b)), D('1e-70'))

    def test_sparse_convolution_matches_full_direct_sum_even_when_taps_overlap(self):
        with localcontext() as context:
            context.prec = 80
            for n in (1, 2, 3, 7, 8):
                for real in (False, True):
                    values, expected = fixtures.convolution_case(n, real)
                    left, right = values[:n], values[n:]
                    direct = []
                    for i in range(n):
                        value = fixtures.ZERO
                        for k in range(n):
                            value = fixtures.add(value, fixtures.mul(left[(i - k) % n], right[k]))
                        direct.append(value)
                    self.assertEqual(expected, direct)

    def test_fixtures_are_reproducible_and_inputs_are_exact_in_float32(self):
        import struct
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary)
            first = fixtures.generate(destination, (1, 3, 8))
            self.assertEqual(first, fixtures.generate(destination, (1, 3, 8)))
            for (value,) in struct.iter_unpack('<d', (destination / 'inputs.bin').read_bytes()):
                self.assertEqual(value, struct.unpack('<f', struct.pack('<f', value))[0])


class ReportTests(unittest.TestCase):
    def test_endpoint_loss_is_visible_and_nonfinite_data_is_rejected(self):
        self.assertEqual(report.compare_values([1 + 0j, 2 + 0j], [1 + 0j, 0j])['max_abs_difference'], 2)
        self.assertFalse(report.compare_values([1j], [complex(float('nan'), 0)])['finite'])
        self.assertIsNone(report.compare_values([0j], [1j])['relative_l2_difference'])

    def test_timing_ratios_never_mix_precision_or_host_and_ipp_is_accuracy_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            directories = []
            for backend, precision, group, duration in (
                    ('pocketfft', 'float', 'mac', 10), ('fftw', 'float', 'mac', 20),
                    ('pocketfft', 'double', 'mac', 100), ('fftw', 'double', 'mac', 200),
                    ('pocketfft', 'float', 'other-host', 1)):
                directory = root / f'{backend}-{precision}-{group}'
                directory.mkdir()
                metadata = dict(system='test', target_architecture='test', execution='native', backend=backend,
                                precision=precision, run_id=group, success=True, reference_sha256='fixture')
                (directory / 'run.json').write_text(json.dumps(metadata))
                rows = [dict(type='summary', completed=1, skipped=0, failures=0),
                        dict(type='timing', n=16, domain='real', operation='forward', median_ns=duration)]
                (directory / 'results.jsonl').write_text('\n'.join(map(json.dumps, rows)))
                (directory / 'outputs.bin').write_bytes(b'')
                directories.append(directory)
            result = report.summarize(directories)
            self.assertEqual([row['time_over_fastest_same_run'] for row in result['timings']], [1, 2, 1, 2, 1])
            changed = copy.deepcopy(json.loads((directories[0] / 'run.json').read_text()))
            changed['backend'] = 'ipp'
            (directories[0] / 'run.json').write_text(json.dumps(changed))
            self.assertTrue(any('IPP/QEMU' in issue for issue in report.summarize(directories)['problems']))


if __name__ == '__main__':
    unittest.main()
