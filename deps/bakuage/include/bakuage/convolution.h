#ifndef BAKUAGE_CONVOLUTION_H_
#define BAKUAGE_CONVOLUTION_H_

#include <cmath>
#include <cstring>
#include <vector>
#ifdef PHASELIMITER_ENABLE_FFTW
#include "fftw3.h"
#include "bakuage/fftw.h"
#endif
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/vector_math.h"

namespace bakuage {

template <typename Float>
void ConvoluteCircular(const Float *x, const Float *y, const int n, Float *output) {
#ifdef PHASELIMITER_ENABLE_FFTW
	double *fft_input;
	fftw_complex *fft_output;
	fftw_complex *fft_output2;
	fftw_plan plan, plan2, plan_inv;
	const int spec_len = n / 2 + 1;

	{
		std::lock_guard<std::recursive_mutex> lock(FFTW::mutex());

		fft_input = (double *)fftw_malloc(sizeof(double) * n);
		std::memset(fft_input, 0, sizeof(double) * n);
		fft_output = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * spec_len);
		std::memset(fft_output, 0, sizeof(fftw_complex) * spec_len);
		fft_output2 = (fftw_complex *)fftw_malloc(sizeof(fftw_complex) * spec_len);
		std::memset(fft_output2, 0, sizeof(fftw_complex) * spec_len);

		plan = fftw_plan_dft_r2c_1d(n, fft_input, fft_output, FFTW_ESTIMATE);
		plan2 = fftw_plan_dft_r2c_1d(n, fft_input, fft_output2, FFTW_ESTIMATE);
		plan_inv = fftw_plan_dft_c2r_1d(n, fft_output, fft_input, FFTW_ESTIMATE);
	}

	double scale = 1.0 / (1e-37 + std::sqrt(n));
	for (int i = 0; i < n; i++) {
		fft_input[i] = x[i] * scale;
	}
	fftw_execute(plan);
	for (int i = 0; i < n; i++) {
		fft_input[i] = y[i] * scale;
	}
	fftw_execute(plan2);

	for (int i = 0; i < spec_len; i++) {
		float fft_output_0 = fft_output[i][0];
		fft_output[i][0] = fft_output_0 * fft_output2[i][0] - fft_output[i][1] * fft_output2[i][1];
		fft_output[i][1] = fft_output_0 * fft_output2[i][1] + fft_output[i][1] * fft_output2[i][0];
	}
	fftw_execute(plan_inv);
	for (int i = 0; i < n; i++) {
		output[i] = fft_input[i];// *scale; // 畳み込みの場合、ここのscaleは不要らしい
	}

	{
		std::lock_guard<std::recursive_mutex> lock(FFTW::mutex());

		fftw_destroy_plan(plan);
		fftw_destroy_plan(plan2);
		fftw_destroy_plan(plan_inv);
		fftw_free(fft_input);
		fftw_free(fft_output);
		fftw_free(fft_output2);
	}
#endif
}

// output: nx + ny - 1
// x, y, outputはアドレスに被りがあっても良い
template <typename Float>
void Convolute(const Float *x, const int nx, const Float *y, const int ny, Float *output) {
#ifdef PHASELIMITER_ENABLE_FFTW
	const int output_len = nx + ny - 1;
	const int output_len_ceil = bakuage::CeilPowerOf2(output_len);
	std::vector<Float> extended_x(output_len_ceil);
	std::vector<Float> extended_y(output_len_ceil);
	std::vector<Float> extended_output(output_len_ceil);

	for (int i = 0; i < nx; i++) {
		extended_x[i] = x[i];
	}
	for (int i = 0; i < ny; i++) {
		extended_y[i] = y[i];
	}
	ConvoluteCircular(extended_x.data(), extended_y.data(), output_len_ceil, extended_output.data());
	for (int i = 0; i < output_len; i++) {
		output[i] = extended_output[i];
	}
#else
    const bool address_overlapped = (output < x + nx && x <= output + nx + ny - 1) || (output < y + ny && y <= output + nx + ny - 1);
    if (address_overlapped) {
        AlignedPodVector<Float> output2(nx + ny - 1);
        VectorConvolve(x, nx, y, ny, output2.data());
        TypedMemcpy(output, output2.data(), nx + ny - 1);
    } else {
        VectorConvolve(x, nx, y, ny, output);
    }
#endif
}

}
#endif
