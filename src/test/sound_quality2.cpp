
#include <random>
#include "gtest/gtest.h"
#include "gflags/gflags.h"
#include "boost/filesystem.hpp"
#include <tbb/tbb.h>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/archive/binary_oarchive.hpp>
#include "bakuage/sound_quality2.h"

DEFINE_string(sound_quality2_analysis_data, "../bakuage_dataset1/analysis", "sound quality2 analysis data dir (bakuage_dataset1/analysis)");
DEFINE_int32(sound_quality2_max_test_data, 0, "max test data count for sound quality2 test");
DEFINE_int32(sound_quality2_cv_folds, 5, "cv folds for sound quality2 test");
DEFINE_int32(sound_quality2_cv_count, 0, "cv count for sound quality2 test");
DEFINE_int32(sound_quality2_worker_count, 0, "worker count for sound quality2 test");

namespace {
    struct Reference {
        Eigen::VectorXd mean;
        Eigen::MatrixXd covariance;
    };
    
    class Stats {
    public:
        void Add(double sound_quality, double base_sound_quality) {
            change_statistics_.Add(sound_quality - base_sound_quality);
            sound_qualities_.push_back(sound_quality);
            for (auto parent: parents_) {
                parent->Add(sound_quality, base_sound_quality);
            }
        }
        void Add(const Stats &other) {
            sound_qualities_.insert(sound_qualities_.end(), other.sound_qualities_.begin(), other.sound_qualities_.end());
            change_statistics_.Add(other.change_statistics_);
            for (auto parent: parents_) {
                parent->Add(other);
            }
        }
        void AddParent(Stats *parent) {
            parents_.push_back(parent);
        }
        
        const std::string summary(const Stats &base_stats) const {
            std::stringstream ss;
            ss << "auc:" << auc(base_stats) << "\tmean:" << change_statistics_.mean() << "\tstddev:" << change_statistics_.stddev();
            return ss.str();
        }
        const double auc(const Stats &base_stats) const {
            return bakuage::CalcAUC(sound_qualities_.begin(), sound_qualities_.end(), base_stats.sound_qualities_.begin(), base_stats.sound_qualities_.end());
        }
        const bakuage::Statistics &change_statistics() const { return change_statistics_; }
    private:
        std::vector<float> sound_qualities_;
        bakuage::Statistics change_statistics_;
        std::vector<Stats *> parents_;
    };
}

TEST(SoundQuality2Calculator, ReferenceVectorIndex) {
    Eigen::VectorXd mean = Eigen::VectorXd::Zero(4);
    Eigen::MatrixXd covariance = Eigen::MatrixXd::Zero(4, 4);
    bakuage::MasteringReference2 reference(mean, covariance);
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(reference.mean_idx(i), i);
    }
    int k = 4;
    for (int i = 0; i < 4; i++) {
        for (int j = i; j < 4; j++) {
            EXPECT_EQ(reference.covariance_idx(i, j), k);
            k++;
        }
    }
}

TEST(SoundQuality2Calculator, SerializationCalculate) {
    using boost::filesystem::recursive_directory_iterator;
    
    std::stringstream ss;
    std::string first_reference_json;
    {
        bakuage::SoundQuality2Calculator calculator;
        recursive_directory_iterator last;
        std::vector<std::string> paths;
        for (recursive_directory_iterator itr("resource/analysis_data"); itr != last; ++itr) {
            const std::string path = itr->path().string();
            if (!bakuage::StrEndsWith(path, ".json")) continue;
            paths.emplace_back(path);
            
            if (first_reference_json.empty()) {
                first_reference_json = bakuage::LoadStrFromFile(path.c_str());
            }
        }
        calculator.PrepareFromPaths(paths.begin(), paths.end());
        boost::archive::binary_oarchive oa(ss);
        oa << calculator;
    }
    
    bakuage::SoundQuality2Calculator calculator2;
    {
        boost::archive::binary_iarchive ia(ss);
        ia >> calculator2;
    }
    
    Eigen::VectorXd mean;
    Eigen::MatrixXd covariance;
    EXPECT_TRUE(bakuage::SoundQuality2CalculatorUnit::ParseReference(first_reference_json.c_str(), &mean, &covariance));

    float sound_quality, lof;
    calculator2.CalculateSoundQuality(mean, covariance, &sound_quality, &lof);
    
    EXPECT_GE(sound_quality, 0);
    EXPECT_LE(sound_quality, 1);
}

// windowsでファイル経由でキャッシュが読み込めないバグの再現テスト
TEST(SoundQuality2Calculator, SerializationFile) {
	using boost::filesystem::recursive_directory_iterator;
	
	const auto temp_path = bakuage::NormalizeToString((boost::filesystem::temp_directory_path() / boost::filesystem::unique_path()).native());
	{
		bakuage::SoundQuality2Calculator calculator;
		recursive_directory_iterator last;
		std::vector<std::string> paths;
		for (recursive_directory_iterator itr("resource/analysis_data"); itr != last; ++itr) {
			const std::string path = itr->path().string();
			if (!bakuage::StrEndsWith(path, ".json")) continue;
			paths.emplace_back(path);
		}
		calculator.PrepareFromPaths(paths.begin(), paths.end());
		std::ofstream ofs(temp_path, std::ios::binary);
		boost::archive::binary_oarchive oa(ofs);
		oa << calculator;
	}

	bakuage::SoundQuality2Calculator calculator2;
	{
		std::ifstream ifs(temp_path, std::ios::binary);
		boost::archive::binary_iarchive ia(ifs);
		ia >> calculator2;
	}
}

// オプションが必要なテスト
// heavy test
TEST(SoundQuality2Calculator, Validation) {
    typedef bakuage::SoundQuality2Calculator Calculator;
    using boost::filesystem::recursive_directory_iterator;
    
    std::vector<Reference> references;
    std::vector<bakuage::SoundQuality2CalculatorUnit::Band> bands;
    {
        recursive_directory_iterator last;
        std::vector<std::string> paths;
        for (recursive_directory_iterator itr(FLAGS_sound_quality2_analysis_data); itr != last; ++itr) {
            const std::string path = itr->path().string();
            if (!bakuage::StrEndsWith(path, ".json")) continue;
            paths.emplace_back(path);
            if (FLAGS_sound_quality2_max_test_data && paths.size() >= FLAGS_sound_quality2_max_test_data) {
                break;
            }
        }
        references.resize(paths.size());
        tbb::parallel_for<int>(0, references.size(), [&paths, &references, &bands](int i) {
            std::ifstream ifs(paths[i].c_str());
            const std::string json_str((std::istreambuf_iterator<char>(ifs)),
                                       std::istreambuf_iterator<char>());
            Eigen::VectorXd mean;
            Eigen::MatrixXd covariance;
            Reference reference;
            EXPECT_TRUE(bakuage::SoundQuality2CalculatorUnit::ParseReference(json_str.c_str(), &reference.mean, &reference.covariance, i == 0 ? &bands : nullptr));
            references[i] = reference;
        });
    }
    
    // データセットを用意
    std::mt19937 engine(1); // seed固定
    std::shuffle(references.begin(), references.end(), engine);
    
    // CVを行う
    const int cv_folds = FLAGS_sound_quality2_cv_folds;
    const int cv_count = FLAGS_sound_quality2_cv_count ? FLAGS_sound_quality2_cv_count : cv_folds;
    Stats total_stats;
    Stats base_stats;
    Stats mid_mean_stats;
    Stats side_mean_stats;
    Stats mid_cov_diag_stats;
    Stats side_cov_diag_stats;
    Stats mid_cov_non_diag_stats;
    Stats other_cov_non_diag_stats;
    mid_mean_stats.AddParent(&total_stats);
    side_mean_stats.AddParent(&total_stats);
    mid_cov_diag_stats.AddParent(&total_stats);
    side_cov_diag_stats.AddParent(&total_stats);
    mid_cov_non_diag_stats.AddParent(&total_stats);
    other_cov_non_diag_stats.AddParent(&total_stats);
    
    // CircleCIでメモリ使いすぎで落ちるので並列数を制限する
    tbb::task_scheduler_init tbb_init(FLAGS_sound_quality2_worker_count ? FLAGS_sound_quality2_worker_count : tbb::task_scheduler_init::default_num_threads());
    for (int cv_i = 0; cv_i < cv_count; cv_i++) {
        // 全てのreference、全ての特徴量に対して、1dB変化させたときのsound_qualityを計算する
        // スペクトル平均、共分散(対角)、共分散(非対角)に分類して統計を取り
        // 平均と標準偏差をvalidationする
        
        const int test_start = (int64_t)cv_i * references.size() / cv_folds;
        const int test_end = (int64_t)(cv_i + 1) * references.size() / cv_folds;
        
        Calculator calculator;
        {
            std::vector<bakuage::MasteringReference2> train_references;
            for (int i = 0; i < references.size(); i++) {
                if (i < test_start || test_end <= i) {
                    train_references.emplace_back(references[i].mean, references[i].covariance);
                }
            }
            calculator.Prepare(train_references.begin(), train_references.end(), bands.begin(), bands.end());
        }
        
        const bakuage::AlignedPodVector<int> changes({-1, 1});
        
        std::mutex mtx;
        tbb::parallel_for(test_start, test_end, [&changes, &references, &calculator, &base_stats, &mid_mean_stats, &side_mean_stats, &mid_cov_diag_stats, &side_cov_diag_stats, &mid_cov_non_diag_stats, &other_cov_non_diag_stats, &mtx](int i) {
            Stats local_base_stats;
            Stats local_mid_mean_stats;
            Stats local_side_mean_stats;
            Stats local_mid_cov_diag_stats;
            Stats local_side_cov_diag_stats;
            Stats local_mid_cov_non_diag_stats;
            Stats local_other_cov_non_diag_stats;
            
            for (int k = 0; k < changes.size(); k++) {
                const bakuage::MasteringReference2 reference(references[i].mean, references[i].covariance);
                float base_sound_quality;
                calculator.CalculateSoundQuality(reference, &base_sound_quality, nullptr);
                local_base_stats.Add(base_sound_quality, base_sound_quality);
                for (int mean_idx = 0; mean_idx < reference.mean_count(); mean_idx++) {
                    auto changed_mean = references[i].mean;
                    changed_mean[mean_idx] += changes[k];
                    const bakuage::MasteringReference2 changed_reference(changed_mean, references[i].covariance);
                    float sound_quality;
                    calculator.CalculateSoundQuality(changed_reference, &sound_quality, nullptr);
                    if (mean_idx % 2 == 0) {
                        local_mid_mean_stats.Add(sound_quality, base_sound_quality);
                    } else {
                        local_side_mean_stats.Add(sound_quality, base_sound_quality);
                    }
                }
                for (int cov_idx1 = 0; cov_idx1 < reference.mean_count(); cov_idx1++) {
                    for (int cov_idx2 = cov_idx1; cov_idx2 < reference.mean_count(); cov_idx2++) {
                        auto changed_cov = references[i].covariance;
                        changed_cov(cov_idx1, cov_idx2) = bakuage::SignedSqr(bakuage::SignedSqrt(changed_cov(cov_idx1, cov_idx2)) + changes[k]);
                        changed_cov(cov_idx2, cov_idx1) = changed_cov(cov_idx1, cov_idx2);
                        const bakuage::MasteringReference2 changed_reference(references[i].mean, changed_cov);
                        float sound_quality;
                        calculator.CalculateSoundQuality(changed_reference, &sound_quality, nullptr);
                        if (cov_idx1 == cov_idx2) {
                            if (cov_idx1 % 2 == 0) {
                                local_mid_cov_diag_stats.Add(sound_quality, base_sound_quality);
                            } else {
                                local_side_cov_diag_stats.Add(sound_quality, base_sound_quality);
                            }
                        } else {
                            if (cov_idx1 % 2 == 0 && cov_idx2 % 2 == 0) {
                                local_mid_cov_non_diag_stats.Add(sound_quality, base_sound_quality);
                            } else {
                                local_other_cov_non_diag_stats.Add(sound_quality, base_sound_quality);
                            }
                        }
                    }
                }
            }
            
            {
                std::lock_guard<std::mutex> lock(mtx);
                base_stats.Add(local_base_stats);
                mid_mean_stats.Add(local_mid_mean_stats);
                side_mean_stats.Add(local_side_mean_stats);
                mid_cov_diag_stats.Add(local_mid_cov_diag_stats);
                side_cov_diag_stats.Add(local_side_cov_diag_stats);
                mid_cov_non_diag_stats.Add(local_mid_cov_non_diag_stats);
                other_cov_non_diag_stats.Add(local_other_cov_non_diag_stats);
            }
        });
    }
    
    EXPECT_GE(mid_mean_stats.auc(base_stats), 0.505);
    EXPECT_LE(mid_mean_stats.change_statistics().mean(), -0.007);
    EXPECT_LE(mid_mean_stats.change_statistics().stddev(), 0.023);
    
    EXPECT_GE(side_mean_stats.auc(base_stats), 0.507);
    EXPECT_LE(side_mean_stats.change_statistics().mean(), -0.009);
    EXPECT_LE(side_mean_stats.change_statistics().stddev(), 0.0283);
    
    EXPECT_GE(mid_cov_diag_stats.auc(base_stats), 0.51);
    EXPECT_LE(mid_cov_diag_stats.change_statistics().mean(), -0.0138);
    EXPECT_LE(mid_cov_diag_stats.change_statistics().stddev(), 0.028);
    
    EXPECT_GE(side_cov_diag_stats.auc(base_stats), 0.516);
    EXPECT_LE(side_cov_diag_stats.change_statistics().mean(), -0.0167);
    EXPECT_LE(side_cov_diag_stats.change_statistics().stddev(), 0.026);
    
    EXPECT_GE(mid_cov_non_diag_stats.auc(base_stats), 0.519);
    EXPECT_LE(mid_cov_non_diag_stats.change_statistics().mean(), -0.0192);
    EXPECT_LE(mid_cov_non_diag_stats.change_statistics().stddev(), 0.039);
    
    EXPECT_GE(other_cov_non_diag_stats.auc(base_stats), 0.524);
    EXPECT_LE(other_cov_non_diag_stats.change_statistics().mean(), -0.0245);
    EXPECT_LE(other_cov_non_diag_stats.change_statistics().stddev(), 0.043);
    
    EXPECT_GE(total_stats.auc(base_stats), 0.521);
    EXPECT_LE(total_stats.change_statistics().mean(), -0.0212);
    EXPECT_LE(total_stats.change_statistics().stddev(), 0.04);
    
    std::cerr << "mid mean\t" << mid_mean_stats.summary(base_stats) << std::endl;
    std::cerr << "side mean\t" << side_mean_stats.summary(base_stats) << std::endl;
    std::cerr << "mid cov_diag\t" << mid_cov_diag_stats.summary(base_stats) << std::endl;
    std::cerr << "side cov_diag\t" << side_cov_diag_stats.summary(base_stats) << std::endl;
    std::cerr << "mid cov_non_diag\t" << mid_cov_non_diag_stats.summary(base_stats) << std::endl;
    std::cerr << "other cov_non_diag\t" << other_cov_non_diag_stats.summary(base_stats) << std::endl;
    std::cerr << "total\t" << total_stats.summary(base_stats) << std::endl;
}
