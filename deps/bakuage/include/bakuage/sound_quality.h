#ifndef bakuage_sound_quality_h
#define bakuage_sound_quality_h

#include <cmath>
#include <vector>
#include <Eigen/Dense>
#include "picojson.h"
#include "bakuage/lof.h"

namespace bakuage {
    
    // private
    struct MasteringReference {
        MasteringReference(const Eigen::VectorXd &_mean, const Eigen::MatrixXd &_covariance): mean(_mean), covariance(_covariance) {
            const int size = mean.size();
            // mean normalization
#if 0
            const double m = mean->mean();
            for (int i = 0; i < size; i++) {
                mean(i) -= m;
            }
#endif
            // regularization
            covariance += 1e-03 * Eigen::MatrixXd::Identity(size, size);
            
            inv_covariance = covariance.inverse();
        }
        
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
        Eigen::MatrixXd inv_covariance;
    };
    
    struct MasteringReferenceDistFunc {
        double operator () (const MasteringReference &a, const MasteringReference &b) {
            return JensenShannonDistance(a.mean, a.covariance, a.inv_covariance, b.mean, b.covariance, b.inv_covariance);
        }
    };
    
    class SoundQualityCalculator {
    public:
        typedef MasteringReferenceDistFunc DistFunc;
        
        SoundQualityCalculator(): lof_(MasteringReferenceDistFunc()), border_lof_(0) {}
        virtual ~SoundQualityCalculator() {}
        
        static bool ParseReference(const char *analysis_json, Eigen::VectorXd *output_mean, Eigen::MatrixXd *output_covariance) {
            using namespace picojson;
            value v;
            std::string err = parse(v, analysis_json);
            if (!err.empty()) {
                std::cerr << "SoundQualityCalculator AddReference error " << err << std::endl;
                return false;
            }
            
            const auto &root = v.get<object>();
            const auto &bands_json = root.at("bands").get<array>();
            Eigen::VectorXd mean(2 * bands_json.size());
            for (int i = 0; i < bands_json.size(); i++) {
                const auto &band_json = bands_json.at(i).get<object>();
                mean(2 * i + 0) = band_json.at("mid_mean").get<double>();
                mean(2 * i + 1) = band_json.at("side_mean").get<double>();
            }
            
            const auto &covariance_json = root.at("covariance").get<array>();
            Eigen::MatrixXd covariance(2 * bands_json.size(), 2 * bands_json.size());
            for (int i = 0; i < 2 * bands_json.size(); i++) {
                const auto row = covariance_json.at(i).get<array>();
                for (int j = 0; j < 2 * bands_json.size(); j++) {
                    covariance(i, j) = row.at(j).get<double>();
                }
            }
            
            *output_mean = mean;
            *output_covariance = covariance;

			return true;
        }
        
        void AddReference(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance) {
            references_.emplace_back(mean, covariance);
        }
        
        bool AddReference(const char *analysis_json) {
            Eigen::VectorXd mean;
            Eigen::MatrixXd covariance;
			if (!ParseReference(analysis_json, &mean, &covariance)) return false;
            AddReference(mean, covariance);
			return true;
        }
        
        void Prepare() {
            lof_.Prepare(references_.begin(), references_.end(), std::ceil(0.03 * references_.size()));
            
            sorted_reference_lofs_.clear();
            for (const auto &reference: references_) {
                const auto lof = lof_.CalculateLof(reference);
                sorted_reference_lofs_.emplace_back(lof);
            }
            std::sort(sorted_reference_lofs_.begin(), sorted_reference_lofs_.end());
            
            border_lof_ = sorted_reference_lofs_[std::floor(sorted_reference_lofs_.size() * 0.9)];
        }
        
        void CalculateSoundQuality(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance, double *output_sound_quality, double *output_lof) {
            MasteringReference reference(mean, covariance);
            const auto lof = lof_.CalculateLof(reference);
            if (output_sound_quality) {
                *output_sound_quality = 1.0 / (1.0 + abs(lof - 1.0) / (border_lof_ - 1.0));
            }
            if (output_lof) {
                *output_lof = lof;
            }
        }
        
        double border_lof() const { return border_lof_; }
    private:
        std::vector<MasteringReference> references_;
        std::vector<double> sorted_reference_lofs_;
        Lof<MasteringReference, double, DistFunc> lof_;
        double border_lof_;
    };
    
}


#endif /* sound_quality_h */
