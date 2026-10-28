import copy
from decimal import Decimal as D, localcontext
import json
from pathlib import Path
import tempfile
import unittest

import fixtures
import peak_reference
import report
import run_peak_probe


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


class PeakReferenceTests(unittest.TestCase):
    def test_probe_requires_all_original_cases_and_detectable_endpoint_controls(self):
        metadata = dict(backends=['pocketfft'], contracts=['off'])
        self.assertTrue(run_peak_probe.validation_problems([], metadata))
        rows = []
        for precision in ('float', 'double'):
            for domain in ('real', 'complex'):
                base = dict(backend='pocketfft', precision=precision, domain=domain, fp_contract='off',
                            scatter_max_error=0)
                rows.append(dict(base, original_case=True, filter='lowpass', fault='none'))
                for fault in ('drop_dc', 'drop_nyquist', 'mix_dc_nyquist'):
                    rows.append(dict(base, original_case=False, filter='endpoints', fault=fault,
                                     product_dc_error=.1, product_nyquist_error=.1, wave_max_abs_error=.1))
        self.assertFalse(run_peak_probe.validation_problems(rows, metadata))
        broken = copy.deepcopy(rows)
        broken[2]['product_nyquist_error'] = 0
        broken[2]['wave_max_abs_error'] = 0
        self.assertTrue(any('not detected' in problem for problem in run_peak_probe.validation_problems(broken, metadata)))
        broken = copy.deepcopy(rows)
        broken[0]['perm_comparison_max_error'] = 1e-7
        self.assertTrue(any('Perm/canonical' in problem for problem in run_peak_probe.validation_problems(broken, metadata)))

    def test_reference_fft_agrees_with_direct_dft(self):
        with localcontext() as context:
            context.prec = 80
            for n in (1, 2, 4, 8, 16, 32):
                values = [(D(i - n // 2) / 16, D(i % 3) / 8) for i in range(n)]
                actual = peak_reference.decimal_fft(values)
                expected = fixtures.direct_dft(values)
                self.assertLess(max(abs(x - y) for a, b in zip(actual, expected)
                                    for x, y in zip(a, b)), D('1e-70'))

    def test_bessel_reference_and_ideal_impulse_peak(self):
        with localcontext() as context:
            context.prec = 80
            self.assertEqual(peak_reference.bessel_i0(D(0)), D(1))
            self.assertLess(abs(peak_reference.bessel_i0(D(1))
                                - D('1.266065877752008335598244625214717537607670311354962206808135331213575016122776')), D('1e-70'))
            fir = peak_reference.ideal_fir()
            self.assertEqual(fir, fir[::-1])
            self.assertEqual(fir[2048], (D('.25'), D(0)))
            self.assertEqual(max(abs(re) * 4 for re, _ in fir), D(1))
            for i, (re, im) in enumerate(fir):
                self.assertEqual(im, 0)
                if i != 2048 and (i - 2048) % 4 == 0:
                    self.assertEqual(re, 0)

    def test_reported_failure_is_exactly_the_float_predecessor_of_one(self):
        import math
        import struct
        value = struct.unpack('<f', struct.pack('<I', 0x3f7fffff))[0]
        db = struct.unpack('<f', struct.pack('<f', 20 * math.log10(value)))[0]
        self.assertEqual(db, -5.1771939979516901e-07)


if __name__ == '__main__':
    unittest.main()
