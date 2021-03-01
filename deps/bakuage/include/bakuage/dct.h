#ifndef BAKUAGE_DCT_H_
#define BAKUAGE_DCT_H_

#include <cmath>
#include <vector>
#include "bakuage/utils.h"

namespace bakuage {
class Dct {
public:
	Dct(int size) : size_(size), size4_(4 * size), cos_table_(4 * size), temp_(size){
		for (int i = 0; i < size4_; i++) {
			cos_table_[i] = std::cos(M_PI / (2 * size_) * i);
		}
	}

	template <class T>
	void DctType2(const T *input, T *output) const {
		for (int k = 0; k < size_; k++) {
			double sum = 0;
			int idx = (2 * 0 + 1) * k;
			int idx_stride = 2 * k;
			for (int n = 0; n < size_; n++) {
				// idx = (2 * n + 1) * k
				sum += input[n] * cos_table_[idx];
				idx = fast_mod_size4(idx + idx_stride);
			}
			output[k] = sum;
		}
	}

	template <class T>
	void DctType2Replacing(T *input_output) {
		std::copy_n(input_output, size_, temp_.begin());
		for (int k = 0; k < size_; k++) {
			double sum = 0;
			int idx = (2 * 0 + 1) * k;
			int idx_stride = 2 * k;
			for (int n = 0; n < size_; n++) {
				// idx = (2 * n + 1) * k
				sum += temp_[n] * cos_table_[idx];
				idx = fast_mod_size4(idx + idx_stride);
			}
			input_output[k] = sum;
		}
	}

	template <class T>
	void DctType3(const T *input, T *output) const {
		for (int k = 0; k < size_; k++) {
			double sum = 0.5 * input[0];
			int idx = 1 * (2 * k + 1);
			int idx_stride = 2 * k;
			for (int n = 1; n < size_; n++) {
				// idx = n * (2 * k + 1)
				sum += input[n] * cos_table_[idx];
				idx = fast_mod_size4(idx + idx_stride);
			}
			output[k] = sum;
		}
	}

	int size() const { return size_; }
private:
	// 0 - 8 * size_ - 1までしか対応しないけど速い
	int fast_mod_size4(int x) const {
		return x >= size4_ ? x - size4_ : x;
	}
	int size_;
	int size4_;
	std::vector<double> cos_table_; // cos(M_PI / (2 * size_) * idx)
	std::vector<double> temp_; // temporary for replacing dct
};
}

#endif
