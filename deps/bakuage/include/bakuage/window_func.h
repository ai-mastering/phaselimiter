
#ifndef BAKUAGE_BAKUAGE_WINDOW_FUNC_H_
#define BAKUAGE_BAKUAGE_WINDOW_FUNC_H_

#include <cmath>
#include "bakuage/bessel.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"

namespace bakuage {   

inline double Keiser(double k, double n, double alpha) {
    return BesselI0(M_PI * alpha * std::sqrt(1 - Sqr(2 * k / n - 1))) / BesselI0(M_PI * alpha);
}
    
#if 0
// バグあり: Keiserを使う場合と結果が食い違う
void CopyKeiserDouble(int n, double alpha, double *it, double scale = 1.0);
    
template <class Iterator>
void CopyKeiser(int n, double alpha, Iterator it, double scale = 1.0) {
    bakuage::AlignedPodVector<double> temp(n);
    CopyKeiserDouble(n, alpha, temp.data(), scale);
    for (int i = 0; i < n; i++) {
        *it = temp[i];
        ++it;
    }
}
#endif

template <class Iterator>
void CopyHanning(int n, Iterator it, double scale = 1.0) {
	const double delta = 2 * M_PI / n;
	for (int i = 0; i < n; i++) {
		*it = std::max<double>(0, 0.5 - 0.5 * std::cos(delta * i)) * scale;
		++it;
	}
}

template <class Iterator>
void CopyBlackmanHarris(int n, Iterator it, double scale = 1.0) {
	const double delta = 2 * M_PI / n;
	for (int i = 0; i < n; i++) {
		const auto theta = delta * i;
		*it = (0.35875 - 0.48829 * std::cos(theta) + 0.14128 * std::cos(2 * theta) - 0.01168 * std::cos(3 * theta)) * scale;
		++it;
	}
}

template <class It1, class It2, class It3>
void ElementWiseMultiply(It1 input1, It2 input2, int n, It3 output) {
	for (int i = 0; i < n; i++) {
		*output = (*input1) * (*input2);
		++input1;
		++input2;
		++output;
	}
}

template <class It1, class It2, class It3>
void ElementWiseAdd(It1 input1, It2 input2, int n, It3 output) {
	for (int i = 0; i < n; i++) {
		*output = (*input1) + (*input2);
		++input1;
		++input2;
		++output;
	}
}

}

#endif 

 
