#ifndef BAKUAGE_NMF_H_
#define BAKUAGE_NMF_H_

#include <algorithm>
#include <random>
#include <Eigen/Dense>

namespace bakuage {

// https://en.wikipedia.org/wiki/Non-negative_matrix_factorization
// Daniel D. Lee and H. Sebastian Seung (2001) Algorithms for Non-negative Matrix Factorization
inline void Nmf(const Eigen::MatrixXd &v, int k, int iter, Eigen::MatrixXd *w, Eigen::MatrixXd *h) {
	*w = Eigen::MatrixXd(v.rows(), k);
	*h = Eigen::MatrixXd(k, v.cols());

	std::mt19937 rand(1);
	std::uniform_real_distribution<double> dist;
	for (int i = 0; i < k; ++i) {
		for (int j = 0; j < v.rows(); j++) {
			(*w)(j, i) = 0.1 + dist(rand);
		}
		for (int j = 0; j < v.cols(); j++) {
			(*h)(i, j) = 0.1 + dist(rand);
		}
	}
    
    Eigen::ArrayXd eps(v.rows() * v.cols());
    for (int i = 0; i < v.rows() * v.cols(); i++) {
        eps(i) = 1e-37;
    }

	// W, Hを更新
	for (int i = 0; i < iter; ++i) {
		w->array() = w->array() * (v * h->transpose()).array() /
			(eps + ((*w) * (*h) * h->transpose()).array());
		h->array() = h->array() * (w->transpose() * v).array() /
			(eps + (w->transpose() * (*w) * (*h)).array());
	}

	// normalize |column of w| = 1
	for (int i = 0; i < w->cols(); ++i) {
		auto norm = w->col(i).norm();
		w->col(i) /= (1e-37 + norm);
		h->row(i) *= norm;
	}

	// std
}
}

#endif 
