#ifndef BAKUAGE_AUDIO_ANALYZER_REVERB_H_
#define BAKUAGE_AUDIO_ANALYZER_REVERB_H_

#include <algorithm>
#include <cmath>
#include <vector>
#include <random>

#include "bakuage/memory.h"
#include "bakuage/dft.h"
#include "bakuage/loudness_filter.h"
#include "bakuage/utils.h"
#include "bakuage/convolution.h"

namespace audio_analyzer {

namespace impl {

// LPC
// https://gist.github.com/hecomi/7398975
//template <typename Float>
//std::vector<Float> CalculateLPCReconstruction(const Float *input, const int N, int order) {
//
//}

// ESTIMATION OF REVERBERATION TIME USING LPC FILTER AND MAXIMUM LIKELIHOOD ESTIMATOR AND DRR
// https://www.ijarcce.com/upload/january/30_ESTIMATION%20OF%20REVERBERATION.pdf
// calc_irがtrueのとき、得られた自己相関が、IRの自己相関だとして、IRを計算する。
// IRは一意に定まらないので、原点が最大になるように選ぶ。

#if 0
template <typename Float>
std::vector<Float> CalculateAutoCorrelation(const Float *input, int samples, bool calc_ir = false) {
	// prepare FFT
	double *fft_input = (double *)fftw_malloc(sizeof(double) * samples);
	std::memset(fft_input, 0, sizeof(double) * samples);
	fftw_complex *fft_output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * (samples / 2 + 1));
	std::memset(fft_output, 0, sizeof(fftw_complex) * (samples / 2 + 1));
	fftw_plan plan = fftw_plan_dft_r2c_1d(samples, fft_input, fft_output, FFTW_ESTIMATE);
	fftw_plan plan_inv = fftw_plan_dft_c2r_1d(samples, fft_output, fft_input, FFTW_ESTIMATE);

	// auto correlation
	double scale = 1.0 / std::sqrt(samples);
	for (int i = 0; i < samples; i++) {
		fft_input[i] = input[i] * scale;
	}
	fftw_execute(plan);
	for (int i = 0; i < samples / 2 + 1; i++) {
		fft_output[i][0] = fft_output[i][0] * fft_output[i][0] + fft_output[i][1] * fft_output[i][1];
		if (calc_ir) {
			fft_output[i][0] = std::sqrt(fft_output[i][0]);
		}
		fft_output[i][1] = 0;
	}

	fftw_execute(plan_inv);
	std::vector<Float> correlation(samples);
	for (int i = 0; i < samples; i++) {
		correlation[i] = fft_input[i] * scale;
	}

	// free FFT
	fftw_destroy_plan(plan);
	fftw_destroy_plan(plan_inv);
	fftw_free(fft_input);
	fftw_free(fft_output);

	return correlation;
}
#endif

/*
LPCはうまくない。なぜかというと、スペクトルがホワイトノイズから遠いときにエネルギーが減るから、
ホワイトノイズに近いか遠いかで重みがばらついてしまう。

スペクトルを均す方法は良いが、下手をするとリバーブ成分をdirectとして扱ってしまうことで、
逆にDRRが下がるみたいなことが起きる。
direct_lengthは重要。
*/
template <typename Float>
void CalculateDirectReverbCore(const Float *input, int samples, int sample_rate, Float *direct, Float *reverb) {
	const int max_length = 0.25 * sample_rate;
	const int direct_length = 0.001 * sample_rate;
	const int dr_transition_length = 0.0001 * sample_rate;
	const int spec_length = samples / 2 + 1;
	const int smooth_window_size = 0 ?
		std::max<int>(1, 1.0 * samples / direct_length) :
		std::max<int>(1, 1.0 * samples / sample_rate * 125);
	const Float scale = 1.0 / std::sqrt(samples);

	// prepare FFT
    float *fft_input = (float *)bakuage::AlignedMalloc(sizeof(float) * samples);
    std::complex<float> *fft_output = (std::complex<float> *)bakuage::AlignedMalloc(sizeof(std::complex<float>) * spec_length);
    bakuage::RealDft<float> dft(samples);

	// whitening by lpc
//	std::vector<Float> lpc_reconstruction = CalculateLPCReconstruction(input, samples, 64);

	// calculate psd
	std::vector<Float> psd(spec_length);
	double energy = 0;
	for (int i = 0; i < samples; i++) {
		energy += input[i] * input[i];
	}
	// const double scale2 = 1.0 / (1e-37 + std::sqrt(energy));
	const int seed = 1;
	std::mt19937 mt(seed);
	std::normal_distribution<> dist(0.0, std::pow(10, -140 / 20.0) / std::sqrt(samples)); // noise db
	for (int i = 0; i < samples; i++) {
		// fprintf(stderr, "%d, %e, %e\n", i, input[i], lpc_reconstruction[i]);
		fft_input[i] = ((input[i] /*- 0 * lpc_reconstruction[i]*/)/* * scale2 + dist(mt)*/) * scale;
	}
    dft.Forward(fft_input, (float *)fft_output);
	for (int i = 0; i < spec_length; i++) {
        psd[i] = std::norm(fft_output[i]);
	}

	// calculate auto correlation
	std::vector<Float> auto_correlation(samples);
	for (int i = 0; i < spec_length; i++) {
        fft_output[i] = std::complex<float>(psd[i] * scale, 0);
	}
    dft.Backward((float *)fft_output, fft_input);
	for (int i = 0; i < samples; i++) {
		auto_correlation[i] = fft_input[i];
	}

	if (0) {
		// normalize psd and auto correlation for calculation accuracy
		Float psd_scale = 1.0 / (1e-37 + auto_correlation[0]);
		for (int i = 0; i < samples; i++) {
			auto_correlation[i] *= psd_scale;
		}
		for (int i = 0; i < spec_length; i++) {
			psd[i] *= psd_scale;
		}
	}

	// calculate smooth psd
	std::vector<Float> smooth_psd(spec_length);
	for (int i = 0; i < spec_length; i++) {
		// ここは畳み込みを意図しているので、あえてscaleをかけない

#if 0
        const float real = i < smooth_window_size ? 1.0 / smooth_window_size : 0;
#else
		const float real = i < smooth_window_size ? (0.5 + 0.5 * std::cos(M_PI * i / smooth_window_size)) / smooth_window_size : 0;
#endif
        fft_output[i] = std::complex<float>(real, 0);
	}
    dft.Backward((float *)fft_output, fft_input);
	for (int i = 0; i < samples; i++) {
		fft_input[i] *= auto_correlation[i] * scale;
	}
	dft.Forward(fft_input, (float *)fft_output);
	for (int i = 0; i < spec_length; i++) {
		if (fft_output[i].real() < 0) {
#if 0
            fprintf(stderr, "%d, %e, %e, %e\n", i, fft_output[i].real(), fft_output[i].imag(), psd[i]);
#endif
		}
		smooth_psd[i] = std::max<Float>(0, fft_output[i].real());
	}

	// calculate whitened IR and decompose direct, reverb
	std::vector<Float> ir_direct(samples);
	std::vector<Float> ir_reverb(samples);
	for (int i = 0; i < spec_length; i++) {
		// fprintf(stderr, "%e, %e\n", psd[i], smooth_psd[i]);
        fft_output[i] = std::complex<float>(std::sqrt(psd[i] / (1e-37 + smooth_psd[i])) * scale, 0);
	}
	dft.Backward((float *)fft_output, fft_input);
	//fprintf(stdout, "x,y\n");
	for (int i = 0; i < samples; i++) {
		const int x = i <= samples / 2 ? i : samples - i;
		//fprintf(stdout, "%d,%e\n", i, fft_input[i]);
		if (i >= 44100) {
			//exit(0);
		}

		if (x < direct_length) {
			ir_direct[i] = fft_input[i];
		}
		else if (x < direct_length + dr_transition_length) {
			const double t = 1.0 * (x - direct_length) / dr_transition_length;
			const double s = 0.5 + 0.5 * std::cos(M_PI * t);
			ir_direct[i] = fft_input[i] * s;
			ir_reverb[i] = fft_input[i] * (1 - s);
		}
		else if (x < max_length) {
			ir_reverb[i] = fft_input[i];
		}
	}

	if (1) {
		// disable unwhiten
		double sum_direct = 0;
		for (int i = 0; i < samples; i++) {
			sum_direct += ir_direct[i] * ir_direct[i];
		}
		*direct = sum_direct;

		double sum_reverb = 0;
		for (int i = 0; i < samples; i++) {
			sum_reverb += ir_reverb[i] * ir_reverb[i];
		}
		*reverb = sum_reverb;
	}
	else {
		// calculate unwhitened direct IR energy
		for (int i = 0; i < samples; i++) {
			fft_input[i] = ir_direct[i] * scale;
		}
		dft.Forward(fft_input, (float *)fft_output);
		double sum_direct = 0;
		for (int i = 0; i < spec_length; i++) {
			const int c = (i == 0 || 2 * i == samples) ? 1 : 2; // 片側FFTなので重複分
            sum_direct += c * bakuage::Sqr(fft_output[i].real()) * smooth_psd[i]; // 虚部 == 0
		}
		*direct = sum_direct;

		// calculate unwhitened reverb IR energy
		for (int i = 0; i < samples; i++) {
			fft_input[i] = ir_reverb[i] * scale;
		}
		dft.Forward(fft_input, (float *)fft_output);
		double sum_reverb = 0;
		for (int i = 0; i < spec_length; i++) {
			const int c = (i == 0 || 2 * i == samples) ? 1 : 2; // 片側FFTなので重複分
			sum_reverb += c * bakuage::Sqr(fft_output[i].real()) * smooth_psd[i]; // 虚部 == 0
		}
		*reverb = sum_reverb;
	}

    bakuage::AlignedFree(fft_input);
    bakuage::AlignedFree(fft_output);
}

}

template <typename Float>
void CalculateDirectReverbEnergyRatio(const Float *input, int channels, int samples, int sample_rate, Float *drr) {
	using namespace impl;

	double sum_reverb = 0;
	double base_line_direct;
	double base_line_reverb;
	std::vector<double> temp(samples);

	// calculate base line (誤差)
	const int seed = 1;
	std::mt19937 mt(seed);
	std::normal_distribution<> dist(0.0, 1.0);
	for (int i = 0; i < samples; i++) {
		temp[i] = dist(mt);
	}
	CalculateDirectReverbCore<double>(temp.data(), samples, sample_rate, &base_line_direct, &base_line_reverb);

	for (int ch = 0; ch < channels; ch++) {
		// main
		for (int i = 0; i < samples; i++) {
			temp[i] = input[i * channels + ch];
		}

		double direct, reverb;
		CalculateDirectReverbCore<double>(temp.data(), samples, sample_rate, &direct, &reverb);

		fprintf(stderr, "%e, %e, %e, %e\n", direct, reverb, base_line_direct, base_line_reverb);

		/*direct = std::max<double>(0, direct - (base_line_direct - 1));
		reverb = std::max<double>(0, reverb - (base_line_reverb - 1));*/
		sum_reverb += reverb / (1e-37 + direct);

	}

	*drr = -10 * std::log10(1e-37 + sum_reverb / channels);
	fprintf(stderr, "drr %f\n", *drr);
}
}

#endif
