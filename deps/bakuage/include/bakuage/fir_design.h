#ifndef BAKUAGE_BAKUAGE_FIR_DESIGN_H_
#define BAKUAGE_BAKUAGE_FIR_DESIGN_H_

#include <cassert>
#include <stdexcept>
#include <vector>
#include "bakuage/window_func.h"
#include "bakuage/utils.h"

namespace bakuage {
    // https://dsp.stackexchange.com/questions/20514/what-is-the-relation-of-the-transition-bands-width-and-the-filter-order-for-the
    // https://www.mathworks.com/help/signal/ug/kaiser-window.html
    // alphaではなくbetaで表記するのがスタンダードなのかも？matlabではbetaらしい
    // e.g. CalcKeiserFirParams(72, 20.0 / 44100, &n, &alpha)
    template <class Float>
    void CalcKeiserFirParams(const double &stopband_reduce_db, const double &normalized_transition_width, int *n, Float *alpha) {
        if (stopband_reduce_db < 21) {
            *alpha = 0;
        } else if (stopband_reduce_db < 50) {
            *alpha = 0.5842 * std::pow(stopband_reduce_db - 21, 0.4) + 0.7886 * (stopband_reduce_db - 21);
        } else {
            *alpha = 0.1102 * (stopband_reduce_db - 8.7);
        }
        *n = bakuage::CeilInt<int>((stopband_reduce_db - 8) / (2.285 * normalized_transition_width), 2) + 1;
    }
    
    // CalculateBandPassFirを複素数に拡張した。
    // http://aidiary.hatenablog.com/entry/20111030/1319895630
    // を参考にかんたんに実装した
    template <class Float>
    std::vector<std::complex<Float>> CalculateBandPassFirComplex(double f1, double f2, int n, double alpha) {
        assert(n % 2 == 1);
        
        // f1 = (f1 - std::floor(f1)) - 0.5;
        // f2 = (f2 - std::floor(f2)) - 0.5;
        
        std::vector<std::complex<Float>> fir;
        for (int i = -(n / 2); i < n / 2 + 1; i++) {
            std::complex<double> im(0, 1);
            std::complex<double> x;
            if (i == 0) {
                x = f2 - f1;
            } else {
                x = (std::exp(2 * M_PI * i * f2 * im) - std::exp(2 * M_PI * i * f1 * im)) / (2 * M_PI * i * im) * Keiser(i + n / 2, n - 1, alpha);
            }
            fir.emplace_back(x);
        }
        assert(fir.size() == n);
        
        return fir;
    }
    
    // DelphiのuFilter.pasのCalcFIR_PF2移植(機能限定版 + 正規化周波数の定義を一般的なものにした)
    // freq1, freq2は正規化周波数
    // alphaの表: http://www.mk.ecei.tohoku.ac.jp/jspmatlab/pdf/matdsp4.pdf
    template <typename Float>
    std::vector<Float> CalculateBandPassFir(Float freq1, Float freq2, int n, Float alpha) {
        using std::sin;
        
        freq1 *= 2;
        freq2 *= 2;

        if (n % 2 != 1) {
            throw std::logic_error("CalculateFir: n must be odd number.");
        }
        if (freq1 < 0 || freq1 > 1) {
            throw std::logic_error("CalculateFir: freq1 must be in [0, 1].");
        }
        if (freq1 > freq2) {
            throw std::logic_error("CalculateFir: freq1 must be <= freq2.");
        }

        int center = (n - 1) / 2;
        static const Float b = 1 / M_PI;

        Float gain1 = 1;
        Float gain2 = 0;

        std::vector<Float> half_fir(center + 1);
#if 1
        half_fir[0] = b * M_PI * ((1 - freq2) * gain2 + (freq2 - freq1) * gain1 + freq1 * gain2);
        for (int i = 1; i <= center; i++) {
            half_fir[i] = b * ((sin(i * M_PI * 1) - sin(i * M_PI * freq2)) * gain2
                + (sin(i * M_PI * freq2) - sin(i * M_PI * freq1)) * gain1 + sin(i * M_PI * freq1) * gain2)
                / i * Keiser(center + i, n - 1, alpha);
        }
#endif
            
        std::vector<Float> result(n);
        // bakuage::CopyKeiser(n, alpha, result.data());
        for (int i = 0; i <= center; i++) {
            result[center - i] = half_fir[i];
            result[center + i] = half_fir[i];
        }
        return result;
    }

	/*template <typename Float>
	std::vector<Float> CalculateInverseFir(const Float *src_fir, int src_fir_n, int dest_fir_n) {
		const int rows = src_fir_n + dest_fir_n - 1;

		Eigen::VectorXd impulse = Eigen::VectorXd::Zero(rows);
		impulse((rows - 1) / 2) = 1;

		Eigen::MatrixXd m = Eigen::MatrixXd::Zero(rows, dest_fir_n);
		for (int src_i = 0; src_i < src_fir_n; src_i++) {
			for (int dest_i = 0; dest_i < dest_fir_n; dest_i++) {
				const row = src_i - dest_i + (dest_fir_n - 1);
				m(row, dest_i) = src_fir[src_i];
			}
		}
		const Eigen::MatrixXd m_inv = m.completeOrthogonalDecomposition().pseudoInverse();

		const Eigen::VectorXd result = m_inv * impulse;
		std::vector<Float> result2(dest_fir_n);
		for (int i = 0; i < dest_fir_n; i++) {
			result2[i] = result(i);
		}
		return result2;
	}*/
}

#endif
