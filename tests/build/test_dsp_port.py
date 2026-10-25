"""Contracts for cross-platform DSP validation and diagnostics."""
import importlib.util
import math
from pathlib import Path
import unittest
from unittest import mock

SCRIPT = Path(__file__).resolve().parents[2] / 'script/check_dsp_port.py'
spec = importlib.util.spec_from_file_location('check_dsp_port', SCRIPT)
dsp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dsp)


def analysis():
    return dict(frames=280049, channels=2, sample_rate=44100, sound_quality2=.35,
                **{key: -12.0 for key in dsp.DB_FIELDS})


class DspPortContracts(unittest.TestCase):
    def test_rounding_tolerance_and_score_regression(self):
        baseline = analysis()
        actual = dict(baseline, loudness=-11.999, sound_quality2=.354)
        self.assertFalse(dsp.compare_analysis(baseline, actual, .1, .005)[1])
        actual['sound_quality2'] = .36
        self.assertTrue(any('sound_quality2' in error for error in
                            dsp.compare_analysis(baseline, actual, .1, .005)[1]))

    def test_audio_shape_and_level_regressions(self):
        baseline = analysis()
        actual = dict(baseline, frames=280048, rms=-11.8)
        errors = dsp.compare_analysis(baseline, actual, .1, .005)[1]
        self.assertTrue(any('frames' in error for error in errors))
        self.assertTrue(any('rms' in error for error in errors))

    def test_nonfinite_values_are_rejected_even_in_nested_analysis(self):
        for value in (math.nan, math.inf, -math.inf):
            actual = dict(analysis(), spectrum=[{'magnitude': value}])
            with self.assertRaisesRegex(ValueError, 'Non-finite'):
                dsp.compare_analysis(analysis(), actual, .1, .005)

    def test_sample_error_and_shape_checks(self):
        self.assertIsNone(dsp.compare_samples([1, -1], [1, -1])['snr_db'])
        result = dsp.compare_samples([1, -1], [.99, -.99])
        self.assertAlmostEqual(result['relative_rms_error'], .01)
        self.assertAlmostEqual(result['snr_db'], 40)
        with self.assertRaisesRegex(ValueError, 'sample count'):
            dsp.compare_samples([1, -1], [1])
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            dsp.compare_samples([1], [math.nan])

    def test_final_diagnostics_and_missing_convergence(self):
        log = ('eval:123 normalized_eval:0.4 last_iter:101 last_iter2:42\n'
               'normalized_eval:0.19 limiting_error:0.21\n')
        self.assertEqual(dsp.convergence(log), dict(normalized_eval=.19, limiting_error=.21,
                                                  iterations1=101, iterations2=42))
        with self.assertRaisesRegex(ValueError, 'Missing final'):
            dsp.convergence('optimization initial_eval: 0.5')
        with self.assertRaisesRegex(ValueError, 'Non-finite'):
            dsp.convergence(log.replace('0.19', 'nan'))

    def test_different_cache_cannot_be_compared(self):
        reference = dict(schema=1, cache_sha256='old', input_sha256={}, cases={})
        actual = dict(reference, cache_sha256='other')
        with self.assertRaisesRegex(ValueError, 'cache_sha256 differs'):
            dsp.compare(reference, actual, Path('.'), Path('.'))

    def test_convergence_regression_fails_but_runner_speed_does_not(self):
        record = dict(seconds=[1.0], analysis=analysis(), audio='out.wav',
                      convergence=dict(normalized_eval=.2, limiting_error=.2,
                                       iterations1=100, iterations2=40))
        baseline = dict(schema=1, cache_sha256='same', input_sha256={}, cases={'limiter': record})
        native_record = dict(record, seconds=[100.0])
        actual = dict(baseline, cases={'limiter': native_record})
        with mock.patch.object(dsp, 'decode_audio', return_value=[1.0, -1.0]):
            self.assertFalse(dsp.compare(baseline, actual, Path('.'), Path('.'))['errors'])
            native_record['convergence'] = dict(record['convergence'], normalized_eval=.3)
            errors = dsp.compare(baseline, actual, Path('.'), Path('.'))['errors']
        self.assertTrue(any('normalized limiter error' in error for error in errors))

    def test_only_linux_to_macos_arm_has_ten_percent_tolerance(self):
        linux = dict(platform='Linux-6.1', architecture='x86_64')
        arm = dict(platform='macOS-15.7.9-arm64', architecture='arm64')
        intel = dict(platform='macOS-15.7.9-x86_64', architecture='x86_64')
        self.assertEqual(.10, dsp.normalized_error_tolerance(linux, arm))
        self.assertEqual(.05, dsp.normalized_error_tolerance(linux, intel))
        self.assertEqual(.05, dsp.normalized_error_tolerance(arm, arm))
        self.assertEqual(.05, dsp.normalized_error_tolerance(arm, linux))

    def test_reported_arm_difference_passes_and_intel_still_fails(self):
        record = dict(seconds=[1.0], analysis=analysis(), audio='out.wav',
                      convergence=dict(normalized_eval=.190574, limiting_error=.20918,
                                       iterations1=101, iterations2=42))
        baseline = dict(schema=2, platform='Linux-6.1', architecture='x86_64',
                        cache_sha256='same', input_sha256={}, cases={'mastering5': record})
        native_record = dict(record, convergence=dict(record['convergence'],
                                                      normalized_eval=.201006, limiting_error=.220345))
        actual = dict(baseline, platform='macOS-15.7.9-arm64', architecture='arm64',
                      cases={'mastering5': native_record})
        with mock.patch.object(dsp, 'decode_audio', return_value=[1.0, -1.0]):
            self.assertFalse(dsp.compare(baseline, actual, Path('.'), Path('.'))['errors'])
            actual['architecture'] = 'x86_64'
            self.assertTrue(dsp.compare(baseline, actual, Path('.'), Path('.'))['errors'])
            actual['architecture'] = 'arm64'
            native_record['convergence']['normalized_eval'] = .211
            self.assertTrue(dsp.compare(baseline, actual, Path('.'), Path('.'))['errors'])

    def test_repeatability_does_not_allow_even_small_numerical_drift(self):
        signature = dict(analysis_sha256='same', normalized_eval=.190574, pcm_sha256='same')
        self.assertFalse(dsp.repeatability_errors('limiter', [signature, dict(signature)]))
        for key, value in [('normalized_eval', .190575), ('pcm_sha256', 'different'),
                           ('analysis_sha256', 'different')]:
            errors = dsp.repeatability_errors('limiter', [signature, dict(signature, **{key: value})])
            self.assertIn(key, errors[0])
            self.assertIn('repeat 1 differs from warmup', errors[0])
        with self.assertRaises(ValueError):
            dsp.repeatability_errors('limiter', [signature])

    def test_repeatability_failures_are_not_hidden_by_cross_platform_tolerance(self):
        baseline = dict(schema=2, cache_sha256='same', input_sha256={}, cases={})
        actual = dict(baseline, repeatability_errors=['limiter: repeat 1 differs from warmup: pcm_sha256'])
        self.assertTrue(dsp.compare(baseline, actual, Path('.'), Path('.'))['errors'])

    def test_convergence_records_normalization_diagnostics(self):
        log = ('unit_eval:58700\nnormalized_eval:0.4 last_iter:101 last_iter2:42\n'
               'unit_eval:121485\nnormalized_eval:0.201006 limiting_error:0.220345\n')
        result = dsp.convergence(log)
        self.assertEqual(121485, result['unit_eval'])
        self.assertAlmostEqual(.201006 * 121485, result['raw_eval_estimate'])
