#ifndef BAKUAGE_STATISTICS_H_
#define BAKUAGE_STATISTICS_H_

#include <cmath>
#include <algorithm>
#include <vector>
#include <limits>
#include <iostream>
#include <Eigen/Dense>

namespace bakuage {

class Statistics {
public:
	Statistics() : count_(0), sum_(0), sum2_(0) {}

	void Add(double value, double count = 1) {
		count_ += count;
		sum_ += value * count;
		sum2_ += value * value * count;
	}
    
    void Add(const Statistics &other) {
        count_ += other.count_;
        sum_ += other.sum_;
        sum2_ += other.sum2_;
    }

	template <class Iterator>
	void AddRange(Iterator bg, Iterator ed, double count = 1) {
		for (Iterator it = bg; it != ed; ++it) {
			Add(*it, count);
		}
	}

	double count() const { return count_; }
	double mean() const { return sum_ / (1e-300 + count_); }
	double sum() const { return sum_; }
	double sum2() const { return sum2_; }
	double variance() const {
		double m = mean();
		return std::max<double>(0.0, sum2_ / (1e-300 + count_) - m * m);
	}
	double stddev() const {
		return std::sqrt(variance());
	}

private:
	friend class StatisticsWithExpFilter;

	double count_;
	double sum_;
	double sum2_;
};


class VectorStatistics {
public:
	VectorStatistics(int dimension) : dimension_(dimension), count_(0), sum_(dimension), 
		min_(dimension, 1e100), max_(dimension, -1e100) {
		for (int i = 0; i < dimension_; i++) {
			sum2_.emplace_back(1 + i);
		}
	}

	template <class RandomAccessIterator>
	void Add(RandomAccessIterator value, double count = 1) {
		count_ += count;
		for (int i = 0; i < dimension_; i++) {
			sum_[i] += value[i] * count;
			min_[i] = std::min<double>(min_[i], value[i]);
			max_[i] = std::max<double>(max_[i], value[i]);
			for (int j = 0; j <= i; j++) {
				sum2_[i][j] += value[i] * value[j] * count;
			}
		}
	}

	// 既存のサンプル数を増減する (指数平均とかに使える)
	void ScaleSamples(double rate) {
		count_ *= rate;
		for (int i = 0; i < dimension_; i++) {
			sum_[i] *= rate;
			for (int j = 0; j <= i; j++) {
				sum2_[i][j] *= rate;
			}
		}
	}

	int dimension() const { return dimension_; }
	double count() const { return count_; }
	std::vector<double> mean() const { 
		std::vector<double> result(dimension_);
		double scale = 1.0 / (1e-300 + count_);
		for (int i = 0; i < dimension_; i++) {
			result[i] = sum_[i] * scale;
		}
		return result;
	}
	const std::vector<double> &sum() const { return sum_; }
	const std::vector<double> &min_vec() const { return min_; }
	const std::vector<double> &max_vec() const { return max_; }
	const std::vector<std::vector<double>> &sum2() const { return sum2_; }
	std::vector<std::vector<double>> covariance() const {
		std::vector<double> mean_vec(mean());
		std::vector<std::vector<double>> result(sum2_);
		double scale = 1.0 / (1e-300 + count_);
		for (int i = 0; i < dimension_; i++) {
			for (int j = 0; j <= i; j++) {
				const double cov = sum2_[i][j] * scale - mean_vec[i] * mean_vec[j];
				if (i == j) {
					result[i][j] = std::max<double>(0, cov);
				}
				else {
					result[i][j] = cov;
				}
			}
		}
		return result;
	}
	// 降順
	std::vector<double> eigen_values() const {
		const int verbose = 0;

		Eigen::MatrixXd m = covariance_as_matrix();

		// https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
		// Singular values are always sorted in decreasing order.
		Eigen::BDCSVD<Eigen::MatrixXd> svd_solver(m, Eigen::ComputeFullU);
		std::vector<double> result(dimension_);
		for (int i = 0; i < dimension_; i++) {
			result[i] = svd_solver.singularValues()(i, 0);
		}
		if (verbose) {
			std::cerr << "eigen values " << svd_solver.singularValues() << std::endl;
			std::cerr << "max eigen vector " << svd_solver.matrixU().col(0) << std::endl;
			std::cerr << "2nd eigen vector " << svd_solver.matrixU().col(1) << std::endl;
		}
		return result;
	}

	Eigen::MatrixXd covariance_as_matrix() const {
		auto cov = covariance();
		Eigen::MatrixXd m(dimension_, dimension_);
		for (int i = 0; i < dimension_; i++) {
			for (int j = 0; j <= i; j++) {
				m(i, j) = m(j, i) = cov[i][j];
			}
		}
		return m;
	}

	double log_determinant(double min_sigma) const {
		auto eigens = eigen_values();
		double result = 0;
		for (int i = 0; i < dimension_; i++) {
			result += std::log(std::max<double>(min_sigma, eigens[i]));
		}
		return result;
	}

private:
	int dimension_;
	double count_;
	std::vector<double> sum_;
	std::vector<std::vector<double>> sum2_;
	std::vector<double> min_;
	std::vector<double> max_;
};

class StatisticsWithExpFilter {
public:
	StatisticsWithExpFilter(double freq): count_(0), sum_(0), sum2_(0), a_(freq) {}

	void Add(double value, double count = 1) {
		auto t = 1 - a_;

		count_ = count_ * t + count * a_;
		sum_ = sum_ * t + value * count * a_;
		sum2_ = sum2_ * t + value * value * count * a_;
	}

	Statistics ToStatistics() const {
		Statistics result;
		result.count_ = count_;
		result.sum_ = sum_;
		result.sum2_ = sum2_;
		return result;
	}

	double count() const { return count_; }
	double mean() const { return sum_ / (1e-300 + count_); }
	double sum() const { return sum_; }
	double sum2() const { return sum2_; }
	double variance() const {
		double m = mean();
		return std::max<double>(0.0, sum2_ / (1e-300 + count_) - m * m);
	}
	double stddev() const {
		return std::sqrt(variance());
	}

private:	
	double count_;
	double sum_;
	double sum2_;
	double a_;
};

}

#endif 
