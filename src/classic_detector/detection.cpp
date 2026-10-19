#include "classic_detector/detector.h"
#include "classic_detector/numpy_neighbors.h"

#include <Eigen/LU>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace classic_detector {
namespace {
// Preserve the old Python metric default for distances absent from its cache.
constexpr double kMetricCovarianceRegularization = 1e-3;
void Validate(const Reference &r) {
    if (!r.mean.size() || r.covariance.rows() != r.mean.size() || r.covariance.cols() != r.mean.size() ||
        !r.mean.allFinite() || !r.covariance.allFinite())
        throw std::invalid_argument("invalid classic reference dimensions or nonfinite values");
}
double RoundedKey(const Reference &r) {
    // np.float64.__round__(7), as used by the Python metric's lookup shortcut.
    return std::nearbyint(r.mean[0] * 1e7) / 1e7;
}
}
double Distance(const Reference &a, const Reference &b, double regularization) {
    Validate(a); Validate(b);
    if (a.mean.size() != b.mean.size() || !std::isfinite(regularization) || regularization < 0)
        throw std::invalid_argument("invalid classic distance parameters");
    Eigen::MatrixXd ca = a.covariance, cb = b.covariance;
    ca.diagonal().array() += regularization; cb.diagonal().array() += regularization;
    Eigen::PartialPivLU<Eigen::MatrixXd> la(ca), lb(cb);
    if ((la.matrixLU().diagonal().array() == 0).any() || (lb.matrixLU().diagonal().array() == 0).any()) throw std::runtime_error("singular classic covariance");
    const Eigen::MatrixXd ia = la.inverse(), ib = lb.inverse();
    const Eigen::VectorXd delta = b.mean - a.mean;
    const double forward = 0.5 * ((ib * ca).trace() + delta.dot(ib * delta) - a.mean.size());
    const double reverse = 0.5 * ((ia * cb).trace() + delta.dot(ia * delta) - a.mean.size());
    const double value = std::sqrt(std::max(0.0, 0.5 * (forward + reverse)));
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite classic distance");
    return value;
}
Detector::Detector(const Preparation &p, const std::vector<Reference> &references,
                   double neighbor_ratio, double outlier_ratio)
    : preparation_(p), references_(references) {
    const size_t n = references.size();
    if (n < 2 || p.paths.size() != n || p.distances.size() != n ||
        !std::isfinite(neighbor_ratio) || neighbor_ratio <= 0 ||
        !std::isfinite(outlier_ratio) || outlier_ratio < 0 || outlier_ratio > 1 ||
        !std::isfinite(p.regularization) || p.regularization < 0)
        throw std::invalid_argument("invalid classic detector configuration (requires at least two references)");
    // sklearn clips n_neighbors to [1, n_samples - 1]. Avoid overflow for huge ratios.
    neighbors_ = static_cast<size_t>(std::max(1.0, std::min<double>(n - 1, std::ceil(neighbor_ratio * n))));
    for (size_t i = 0; i < n; ++i) {
        Validate(references[i]);
        if (references[i].mean.size() != references[0].mean.size() || p.distances[i].size() != n)
            throw std::invalid_argument("inconsistent classic preparation dimensions");
        for (double distance : p.distances[i]) if (!std::isfinite(distance) || distance < 0)
            throw std::invalid_argument("invalid preparation distance");
        const auto inserted = index_by_key_.emplace(RoundedKey(references[i]), static_cast<int>(i));
        if (!inserted.second) inserted.first->second = -1;
    }
    std::vector<std::vector<size_t>> neighbors(n);
    std::vector<std::vector<double>> distances(n);
    k_distances_.resize(n); densities_.resize(n); lofs_.resize(n);
    for (size_t i = 0; i < n; ++i) {
        std::vector<double> row(n);
        for (size_t j = 0; j < n; ++j) {
            // pairwise_distances(callable) computes the upper triangle and mirrors it.
            row[j] = i <= j ? Metric(references[i], references[j]) : Metric(references[j], references[i]);
        }
        auto selected = NumpyNeighbors(row, neighbors_ + 1);
        auto self = std::find(selected.begin(), selected.end(), i);
        selected.erase(self == selected.end() ? selected.begin() : self);
        neighbors[i] = selected;
        for (size_t j : selected) distances[i].push_back(row[j]);
        k_distances_[i] = distances[i].back();
    }
    for (size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (size_t j = 0; j < neighbors_; ++j) sum += std::max(distances[i][j], k_distances_[neighbors[i][j]]);
        densities_[i] = 1 / (sum / neighbors_ + 1e-10);
    }
    for (size_t i = 0; i < n; ++i) {
        double sum = 0;
        for (size_t j : neighbors[i]) sum += densities_[j] / densities_[i];
        lofs_[i] = sum / neighbors_;
    }
    // -scoreatpercentile(negative_outlier_factor_, 100 * outlier_ratio),
    // scipy 1.0.1's default fractional interpolation.
    std::vector<double> negative(n);
    for (size_t i = 0; i < n; ++i) negative[i] = -lofs_[i];
    std::sort(negative.begin(), negative.end());
    const double position = (n - 1) * outlier_ratio;
    const size_t low = static_cast<size_t>(position), high = static_cast<size_t>(std::ceil(position));
    const double fraction = position - low;
    border_ = -(negative[low] * (1 - fraction) + negative[high] * fraction);
}
int Detector::Lookup(const Reference &r) const {
    const auto found = index_by_key_.find(RoundedKey(r));
    return found == index_by_key_.end() ? -1 : found->second;
}
double Detector::Metric(const Reference &a, const Reference &b) const {
    const int i = Lookup(a), j = Lookup(b);
    if (i >= 0 && j >= 0) return preparation_.distances[i][j];
    // The old metric uses its fixed default, not preparation.regularization.
    return Distance(a, b, kMetricCovarianceRegularization);
}
Detection Detector::Detect(const Reference &source) const {
    Validate(source);
    if (source.mean.size() != references_[0].mean.size()) throw std::invalid_argument("source/reference dimensions differ");
    std::vector<double> distances(references_.size());
    for (size_t i = 0; i < references_.size(); ++i) distances[i] = Metric(source, references_[i]);
    const auto neighbors = NumpyNeighbors(distances, neighbors_);
    double reach = 0;
    for (size_t j : neighbors) reach += std::max(distances[j], k_distances_[j]);
    const double density = 1 / (reach / neighbors_ + 1e-10);
    double lof = 0;
    for (size_t j : neighbors) lof += densities_[j] / density;
    lof /= neighbors_;
    size_t selected = references_.size();
    double best = 0;
    for (size_t i = 0; i < references_.size(); ++i) if (lofs_[i] <= border_) {
        // Ranking deliberately bypasses the rounded-key metric shortcut.
        const double distance = Distance(source, references_[i], preparation_.regularization);
        if (selected == references_.size() || distance < best) { selected = i; best = distance; }
    }
    if (selected == references_.size()) throw std::runtime_error("classic detector has no eligible references");
    return {lof, border_, 1 / (1 + std::abs(lof - 1) / (border_ - 1)), selected, lofs_[selected]};
}
}
