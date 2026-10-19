#ifndef CLASSIC_DETECTOR_DETECTOR_H_
#define CLASSIC_DETECTOR_DETECTOR_H_

#include <Eigen/Core>
#include <string>
#include <unordered_map>
#include <vector>

namespace classic_detector {
// Short-window means are interleaved Mid/Side, in dB relative to loudness.
struct Reference {
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
};
struct Preparation {
    double regularization = 1e-3;
    std::vector<std::string> paths;
    std::vector<std::vector<double>> distances;
};
struct Detection {
    double lof;
    double border_lof;
    double sound_quality;
    size_t index;
    double reference_lof;
};
Reference ReadReference(const std::string &path);
Preparation ReadPreparation(const std::string &path);
// Prefer the dataset next to the preparation, including legacy resource/ paths.
// Fall back to the original working-directory-relative path for old datasets.
std::string ResolveReferencePath(const std::string &recorded, const std::string &preparation_path);
std::string DetectionJson(const Detection &detection, const Preparation &preparation);
// sqrt of half the symmetric Gaussian KL (historically named Jensen-Shannon).
double Distance(const Reference &a, const Reference &b, double regularization);

// Brute-force sklearn 0.19.1 LOF, with NumPy 1.14.3 neighbor tie ordering.
// The legacy metric uses a fixed regularization, separately from preparation.
class Detector {
public:
    Detector(const Preparation &preparation, const std::vector<Reference> &references,
             double neighbor_ratio = 0.03, double outlier_ratio = 0.1);
    Detection Detect(const Reference &source) const;
    const std::vector<double> &reference_lofs() const { return lofs_; }
private:
    Preparation preparation_;
    std::vector<Reference> references_;
    std::unordered_map<double, int> index_by_key_;
    std::vector<double> k_distances_, densities_, lofs_;
    size_t neighbors_;
    double border_;
    int Lookup(const Reference &reference) const;
    double Metric(const Reference &a, const Reference &b) const;
};
}
#endif
