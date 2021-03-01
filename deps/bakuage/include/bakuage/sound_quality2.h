#ifndef bakuage_sound_quality2_h
#define bakuage_sound_quality2_h

#include <cmath>
#include <array>
#include <list>
#include <vector>
#include <Eigen/Dense>
#ifndef BAKUAGE_DISABLE_TBB
#include <tbb/tbb.h>
#endif
#include "picojson.h"
#include "bakuage/lof.h"
#include "bakuage/vector_math.h"
#include "bakuage/statistics.h"
#include "bakuage/eigen_serialization.h"

// #define BA_SOUND_QUALITY2_KL
// #define BA_SOUND_QUALITY2_MEAN_COV_DIAG_ONLY

namespace bakuage {
    struct MasteringReference2 {
        struct DistFunc {
            // PCAをやるので回転不変なL2が良いと思う
            float operator () (const MasteringReference2 &a, const MasteringReference2 &b) const {
#ifdef BA_SOUND_QUALITY2_KL
                return JensenShannonDistance(a.mean_, a.covariance_, a.inv_covariance_, b.mean_, b.covariance_, b.inv_covariance_);
#else
                return bakuage::VectorNormDiffL2(a.vec_.data(), b.vec_.data(), a.vec_.size());
#endif
            }
        private:
            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version) {}
        };
        
        MasteringReference2(): mean_count_(0) {}
        
        MasteringReference2(const Eigen::VectorXd &_mean, const Eigen::MatrixXd &_covariance): mean_count_(_mean.size()), vec_(_mean.size() + (_mean.size() + 1) * _mean.size() / 2) {
            for (int i = 0; i < mean_count_; i++) {
                vec_[mean_idx(i)] = _mean[i];
            }
            for (int i = 0; i < mean_count_; i++) {
                for (int j = i; j < mean_count_; j++) {
#if 1
                    const auto x = bakuage::SignedSqrt(_covariance(i, j));
#else
                    const auto x = _covariance(i, j);
#endif
                    vec_[covariance_idx(i, j)] = x;
                }
            }
            
#ifdef BA_SOUND_QUALITY2_KL
            mean_ = _mean.cast<float>();
            // regularization (正則化項を1くらいにすると、meanの違いを見分けられなくなる。ゼロに近いモノラルとかを見分けているだけ？)
            // 1e-7とかにしても1e-3と傾向変わらず
            const auto reg = _covariance + 1e-3 * Eigen::MatrixXd::Identity(mean_count_, mean_count_);
            covariance_ = reg.cast<float>();
            inv_covariance_ = reg.inverse().cast<float>();
#else
            // rms normalization
            double ene = 0;
            for (int i = 0; i < mean_count_; i++) {
                ene += std::pow(10, vec_[mean_idx(i)] / 10);
            }
            for (int i = 0; i < mean_count_; i++) {
                vec_[mean_idx(i)] = 10 * std::log10(std::pow(10, vec_[mean_idx(i)] / 10) / (1e-37 + ene));
            }
#endif
        }
        
        void ResizeMeanOnly() {
            vec_.resize(mean_count_);
        }
        
#ifdef BA_SOUND_QUALITY2_MEAN_COV_DIAG_ONLY
        void ResizeMeanCovDiagOnly() {
            for (int i = 0; i < mean_count_; i++) {
                vec_[mean_count_ + i] = vec_[covariance_idx(i, i)];
            }
            vec_.resize(2 * mean_count_);
        }
#endif
        
        int mean_count() const { return mean_count_; }
        int mean_idx(int i) const { return i; }
        int covariance_idx(int i, int j) const { return mean_count_ + (2 * mean_count_ - (i - 1)) * i / 2 + j - i; }
        int vec_size() const { return vec_.size(); }
        float *vec() { return vec_.data(); }
        
#ifdef BA_SOUND_QUALITY2_KL
        Eigen::VectorXf mean_;
        Eigen::MatrixXf covariance_;
        Eigen::MatrixXf inv_covariance_;
#endif
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & mean_count_;
            ar & vec_;
#ifdef BA_SOUND_QUALITY2_KL
            ar & mean_;
            ar & covariance_;
            ar & inv_covariance_;
#endif
        }
        int mean_count_;
        bakuage::AlignedPodVector<float> vec_;
    };
    
    class SoundQuality2CalculatorUnit {
    public:
        typedef typename MasteringReference2::DistFunc DistFunc;
        
        struct Band {
            float low_freq;
            float high_freq;
            
        private:
            friend class boost::serialization::access;
            template<class Archive>
            void serialize(Archive & ar, const unsigned int version) {
                ar & low_freq;
                ar & high_freq;
            }
        };
        
        enum Mode {
            kModeFull,
            kModeMeanOnly,
        };
        
        SoundQuality2CalculatorUnit(Mode mode): mode_(mode), lof_(DistFunc()) {}
        virtual ~SoundQuality2CalculatorUnit() {}
        
        static bool ParseReference(const char *analysis_json, Eigen::VectorXd *output_mean, Eigen::MatrixXd *output_covariance, std::vector<Band> *bands = nullptr) {
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
            
            if (bands) {
                bands->resize(bands_json.size());
                for (int i = 0; i < bands_json.size(); i++) {
                    const auto &band = bands_json.at(i).get<object>();
                    if (band.find("low_freq") == band.end()) {
                        (*bands)[i].low_freq = 0;
                    }
                    else {
                        (*bands)[i].low_freq = band.at("low_freq").get<double>();
                    }
                    if (band.find("high_freq") == band.end()) {
                        (*bands)[i].high_freq = 0;
                    }
                    else {
                        (*bands)[i].high_freq = band.at("high_freq").get<double>();
                    }
                }
            }
            
            *output_mean = mean;
            *output_covariance = covariance;
            
            return true;
        }
        
#ifndef BAKUAGE_DISABLE_TBB
        template <class ReferenceIt, class BandIt>
        void Prepare(ReferenceIt reference_bg, ReferenceIt reference_ed, BandIt band_bg, BandIt band_ed) {
            std::vector<MasteringReference2> references(reference_bg, reference_ed);
            bands_.clear();
            bands_.insert(bands_.begin(), band_bg, band_ed);
            
#ifndef BA_SOUND_QUALITY2_KL
            // 微調整
            for (int i = 0; i < references.size(); i++) {
                NormalizeReference(&references[i]);
            }
            
            // standard scaler
            standard_scaler_shifts_.resize(references[0].vec_size());
            standard_scaler_scales_.resize(references[0].vec_size());
            if (true) {
                std::vector<bakuage::Statistics> statistics(references[0].vec_size());
                for (int i = 0; i < references.size(); i++) {
                    for (int j = 0; j < references[0].vec_size(); j++) {
                        statistics[j].Add(references[i].vec()[j]);
                    }
                }
                for (int j = 0; j < references[0].vec_size(); j++) {
                    standard_scaler_shifts_[j] = -statistics[j].mean();
                    standard_scaler_scales_[j] = 1.0 / (1e-37 + statistics[j].stddev());
                }
                // 重み調整
                for (int j = 0; j < references[0].mean_count(); j++) {
                    // 両方有効にすると成績少し上がる
#if 1
                    standard_scaler_scales_[references[0].mean_idx(j)] *= std::sqrt(references[0].mean_count());
#endif
#if 1
                    standard_scaler_scales_[references[0].covariance_idx(j, j)] *= std::sqrt(references[0].mean_count());
#endif
                }
                for (int i = 0; i < references.size(); i++) {
                    for (int j = 0; j < references[0].vec_size(); j++) {
                        references[i].vec()[j] = (references[i].vec()[j] + standard_scaler_shifts_[j]) * standard_scaler_scales_[j];
                    }
                }
            } else {
                std::fill_n(standard_scaler_scales_.begin(), standard_scaler_scales_.size(), 1);
            }
            
            if (mode_ == kModeMeanOnly) {
                for (int i = 0; i < references.size(); i++) {
                    references[i].ResizeMeanOnly();
                }
            }
            
#ifdef BA_SOUND_QUALITY2_MEAN_COV_DIAG_ONLY
            for (int i = 0; i < references.size(); i++) {
                references[i].ResizeMeanCovDiagOnly();
                for (int j = 0; j < references[i].mean_count(); j++) {
                    standard_scaler_shifts_[references[i].mean_count() + j] = standard_scaler_shifts_[references[i].covariance_idx(j, j)];
                    standard_scaler_scales_[references[i].mean_count() + j] = standard_scaler_scales_[references[i].covariance_idx(j, j)];
                }
            }
#endif
            
            // pca + whiten
            if (true) {
                Eigen::MatrixXd m(references.size(), references[0].vec_size());
                for (int i = 0; i < references.size(); i++) {
                    for (int j = 0; j < references[0].vec_size(); j++) {
                        m(i, j) = references[i].vec()[j];
                    }
                }
                
                // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
                // Singular values are always sorted in decreasing order.
                const Eigen::BDCSVD<Eigen::MatrixXd> svd_solver(m, Eigen::ComputeFullU | Eigen::ComputeFullV);
                for (int i = 0; i < references.size(); i++) {
                    for (int j = 0; j < references[0].vec_size(); j++) {
                        references[i].vec()[j] = svd_solver.matrixU()(i, j);
                    }
                }
                Eigen::VectorXd inv_singular_values = (1e-37 * Eigen::VectorXd::Ones(references[0].vec_size()) + svd_solver.singularValues()).cwiseInverse();
#if 0
                for (int i = references[0].mean_count(); i < references[0].vec_size(); i++) {
                    inv_singular_values[i] = 0;
                }
#endif
                pca_mat_ = inv_singular_values.asDiagonal() * svd_solver.matrixV().transpose();
            } else {
                pca_mat_ = Eigen::MatrixXd::Identity(references[0].vec_size(), references[0].vec_size());
            }
#endif
            
            // LOF
            lof_.Prepare(references.begin(), references.end(), std::ceil(0.03 * references.size()));
            
            // Quantile Transformer
            sorted_reference_lofs_.resize(references.size());
            tbb::parallel_for<int>(0, references.size(), [this, &references](int i) {
                sorted_reference_lofs_[i] = lof_.CalculateLof(references[i]);
            });
            std::sort(sorted_reference_lofs_.begin(), sorted_reference_lofs_.end());
        }
#endif
        
        double CalculateDistance(const MasteringReference2 &reference, const MasteringReference2 &target) const {
            MasteringReference2 proprocessed = reference;
            MasteringReference2 proprocessed_target = target;
#ifndef BA_SOUND_QUALITY2_KL
            PreprocessReference(&proprocessed);
            PreprocessReference(&proprocessed_target);
#endif
            DistFunc dist_func;
            return dist_func(proprocessed, proprocessed_target);
        }
        
        void CalculateSoundQuality(const MasteringReference2 &reference, float *output_sound_quality, float *output_lof) const {
            MasteringReference2 proprocessed = reference;
#ifndef BA_SOUND_QUALITY2_KL
            PreprocessReference(&proprocessed);
#endif
            const auto lof = lof_.CalculateLof(proprocessed);
            if (output_sound_quality) {
                auto it = std::lower_bound(sorted_reference_lofs_.begin(), sorted_reference_lofs_.end(), lof);
                int pos = std::distance(sorted_reference_lofs_.begin(), it);
                *output_sound_quality = 1.0 - 1.0 * pos / sorted_reference_lofs_.size();
            }
            if (output_lof) {
                *output_lof = lof;
            }
        }
        
        void CalculateSoundQuality(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance, float *output_sound_quality, float *output_lof) const {
            MasteringReference2 reference(mean, covariance);
            CalculateSoundQuality(reference, output_sound_quality, output_lof);
        }
        
        int band_count() const { return bands_.size(); }
        const Band *bands() const { return bands_.data(); }
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & mode_;
            ar & sorted_reference_lofs_;
            ar & lof_;
            ar & bands_;
#ifndef BA_SOUND_QUALITY2_KL
            ar & standard_scaler_shifts_;
            ar & standard_scaler_scales_;
            ar & pca_mat_;
#endif
        }
        
#ifndef BA_SOUND_QUALITY2_KL
        void NormalizeReference(MasteringReference2 *reference) const {
#if 0
            // 成績が少し上がる
            // 1dB変化時のsound quality変化量のstddevを少し上げる
            for (int i = 0; i < reference->mean_count(); i++) {
                reference->vec()[reference->mean_idx(i)] /= 1e-1 + reference->vec()[reference->covariance_idx(i, i)];
            }
#endif
#if 0
            // 成績が少し上がる
            // 1dB変化時のsound quality変化量のstddevをかなり上げてしまう
            for (int i = 0; i < reference->mean_count(); i++) {
                for (int j = i + 1; j < reference->mean_count(); j++) {
                    reference->vec()[reference->covariance_idx(i, j)] /= 1e-1 + std::sqrt(reference->vec()[reference->covariance_idx(i, i)] * reference->vec()[reference->covariance_idx(j, j)]);
                }
            }
#endif
        }
        
        void PreprocessReference(MasteringReference2 *reference) const {
            NormalizeReference(reference);
            
            if (mode_ == kModeMeanOnly) {
                reference->ResizeMeanOnly();
            }
            
#ifdef BA_SOUND_QUALITY2_MEAN_COV_DIAG_ONLY
            reference->ResizeMeanCovDiagOnly();
#endif
            
            Eigen::VectorXd vec(reference->vec_size());
            for (int j = 0; j < reference->vec_size(); j++) {
                vec[j] = (reference->vec()[j] + standard_scaler_shifts_[j]) * standard_scaler_scales_[j];
            }
            vec = pca_mat_ * vec;
            for (int j = 0; j < reference->vec_size(); j++) {
                reference->vec()[j] = vec[j];
            }
        }
#endif
        Mode mode_;
        
        bakuage::AlignedPodVector<float> sorted_reference_lofs_;
        Lof<MasteringReference2, float, DistFunc> lof_;
        std::vector<Band> bands_;
        
        // preprocess
        bakuage::AlignedPodVector<float> standard_scaler_shifts_;
        bakuage::AlignedPodVector<float> standard_scaler_scales_;
        Eigen::MatrixXd pca_mat_;
    };
    
    class SoundQuality2Calculator {
    public:
        SoundQuality2Calculator(): full_unit_(SoundQuality2CalculatorUnit::kModeFull), mean_only_unit_(SoundQuality2CalculatorUnit::kModeMeanOnly) {
            units_[0] = &full_unit_;
            units_[1] = &mean_only_unit_;
        }
        virtual ~SoundQuality2Calculator() {}
        
#ifndef BAKUAGE_DISABLE_TBB
        template <class It>
        void PrepareFromPaths(It path_bg, It path_ed) {
            std::vector<std::string> paths(path_bg, path_ed);
            std::vector<MasteringReference2> references(paths.size());
            std::vector<SoundQuality2CalculatorUnit::Band> bands;
            tbb::parallel_for<int>(0, references.size(), [&paths, &references, &bands](int i) {
                const auto json_str = bakuage::LoadStrFromFile(paths[i].c_str());
                Eigen::VectorXd mean;
                Eigen::MatrixXd covariance;
                if (!SoundQuality2CalculatorUnit::ParseReference(json_str.c_str(), &mean, &covariance, i == 0 ? &bands : nullptr)) {
                    throw std::logic_error(std::string("failed to parse analysis json ") + paths[i]);
                }
                MasteringReference2 reference(mean, covariance);
                references[i] = reference;
            });
            Prepare(references.begin(), references.end(), bands.begin(), bands.end());
        }
#endif
        
#ifndef BAKUAGE_DISABLE_TBB
        template <class ReferenceIt, class BandIt>
        void Prepare(ReferenceIt reference_bg, ReferenceIt reference_ed, BandIt band_bg, BandIt band_ed) {
            std::vector<MasteringReference2> references(reference_bg, reference_ed);
            
            tbb::parallel_for<int>(0, units_.size(), [this, &references, band_bg, band_ed](int i) {
                units_[i]->Prepare(references.begin(), references.end(), band_bg, band_ed);
            });
            
            // Quantile Transformer
            sorted_reference_lofs_.resize(references.size());
            tbb::parallel_for<int>(0, references.size(), [this, &references](int i) {
                CalculateSoundQuality(references[i], nullptr, &sorted_reference_lofs_[i]);
            });
            std::sort(sorted_reference_lofs_.begin(), sorted_reference_lofs_.end());
        }
#endif
        
        double CalculateDistance(const MasteringReference2 &reference, const MasteringReference2 &target) const {
            double distance = 0;
            for (const auto &unit: units_) {
                distance += unit->CalculateDistance(reference, target);
            }
            return distance;
        }
        
        void CalculateSoundQuality(const MasteringReference2 &reference, float *output_sound_quality, float *output_lof) const {
            
            double mean_lof = 0;
            for (const auto &unit: units_) {
                float sq, lof;
                unit->CalculateSoundQuality(reference, &sq, &lof);
#if 0
#if 1
                mean_lof += std::pow(10, 1 - sq);
#else
                mean_lof = std::max<double>(mean_lof, -sq);
#endif
#else
                mean_lof += -sq;
#endif
            }
#if 0
            mean_lof = std::log(mean_lof / 2) / std::log(10);
#endif
#if 1
            mean_lof /= units_.size();
#endif
            
            if (output_sound_quality) {
                auto it = std::lower_bound(sorted_reference_lofs_.begin(), sorted_reference_lofs_.end(), mean_lof);
                int pos = std::distance(sorted_reference_lofs_.begin(), it);
                *output_sound_quality = 1.0 - 1.0 * pos / sorted_reference_lofs_.size();
            }
            if (output_lof) {
                *output_lof = mean_lof;
            }
        }
        
        void CalculateSoundQuality(const Eigen::VectorXd &mean, const Eigen::MatrixXd &covariance, float *output_sound_quality, float *output_lof) const {
            MasteringReference2 reference(mean, covariance);
            CalculateSoundQuality(reference, output_sound_quality, output_lof);
        }
        
        int band_count() const { return units_[0]->band_count(); }
        const SoundQuality2CalculatorUnit::Band *bands() const { return units_[0]->bands(); }
    private:
        friend class boost::serialization::access;
        template<class Archive>
        void serialize(Archive & ar, const unsigned int version) {
            ar & sorted_reference_lofs_;
            ar & full_unit_;
            ar & mean_only_unit_;
        }
        
        bakuage::AlignedPodVector<float> sorted_reference_lofs_;
        bakuage::SoundQuality2CalculatorUnit full_unit_;
        bakuage::SoundQuality2CalculatorUnit mean_only_unit_;
        std::array<bakuage::SoundQuality2CalculatorUnit *, 2> units_;
    };
}


#endif /* sound_quality_h */

