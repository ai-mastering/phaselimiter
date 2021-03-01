#ifndef BAKUAGE_MASTERING3_SCORE_H_
#define BAKUAGE_MASTERING3_SCORE_H_

#include <functional>
#include <vector>
#include <iostream>
#include <algorithm>
#include <numeric>
#include "bakuage/mfcc.h"
#include "bakuage/memory.h"
#include "bakuage/statistics.h"
#include "bakuage/utils.h"
#include "bakuage/nmf.h"
#include "bakuage/loudness_contours.h"

namespace bakuage {

// MFCCは、mfcc[time * band_count + band_index]
// mfccはenergy sum modeを想定している
template <class SpeakerCompensationFunc, class Func2, class Func3>
void CalculateMastering3Score(const float *mid_mel_bands, const float *side_mel_bands,
	const float *noise_mel_bands,
	int count, const bakuage::MfccCalculator<float> &mfcc_calculator, const double &target_sn_db,
	const SpeakerCompensationFunc &speaker_compensation, // loudnessにも影響する
	const Func2 &acoustic_entropy_mid_compensation, // loudnessには影響しない
	const Func3 &acoustic_entropy_side_compensation, // loudnessには影響しない
	float *loudness, float *ear_damage, float *acoustic_entropy_mfcc, float *acoustic_entropy_eigen, float *diff_acoustic_entropy_eigen) {

	const int band_count = mfcc_calculator.num_filters();
	const int verbose2 = 0;

	// サブ指標: 近似ラウドネス
	// Mid, Side各mel bandのエネルギーを重み付けして合計してdBに変換

	// サブ指標: 耳ダメージ (各mel)
	// Mid, Side各mel bandのエネルギーを重み付けしてdBに変換してmax

	// サブ指標: ノイズ下Acoustic Entropy
	// Mid, Side各mel bandをノイズでsaturationして、mfccにして、分散を計算し、shrinkageし、sum (log)
	// ノイズレベルは耳ダメージから求める。

	// 単位はエネルギーの比 (gain dBをlinearに変換したもの)
    bakuage::AlignedPodVector<float> speaker_weights(band_count);
	bakuage::AlignedPodVector<float> loudness_weights(band_count);
	bakuage::AlignedPodVector<float> ear_damage_weights(band_count);
	bakuage::AlignedPodVector<float> acoustic_entropy_mid_weights(band_count);
	bakuage::AlignedPodVector<float> acoustic_entropy_side_weights(band_count);
	for (int j = 0; j < band_count; j++) {
		auto freq = mfcc_calculator.center_freq(j);
		speaker_weights[j] = std::pow(10, speaker_compensation(freq) * 0.1);
		acoustic_entropy_mid_weights[j] = std::pow(10, acoustic_entropy_mid_compensation(freq) * 0.1);
		acoustic_entropy_side_weights[j] = std::pow(10, acoustic_entropy_side_compensation(freq) * 0.1);
		loudness_weights[j] = std::pow(10, 
			(bakuage::loudness_contours::HzToSplAt60Phon(1000) - bakuage::loudness_contours::HzToSplAt60Phon(freq)) * 0.1);
		ear_damage_weights[j] = 
			//1.0 + freq / 100.0; // なんとなく
			//(1.0 + freq / 1000.0) * (1.0 + 1.0 / (freq / 1000.0));
			std::pow(10, (3.0 * std::log2(freq / 1000.0)) * 0.1);

		// high gain: 1.0 + freq / 1000
		// low gain: 1.0 + 1 / (freq / 1000)
	}

	// 近似ラウドネスと耳ダメージ (ついでに、ノイズの基準ラウドネスも)
	// ear_damage_max_modeはfalseのほうが良いかも。maxだと最適化が難しくなってロバストでなくなりそう
	// でもear_damage_max_modeがfalseだと、ear_damage_weights / loudness_weightsが最大のところだけをブーストしまくるのが最適解になる。
	// それは違う。各帯域、痛くないレベルまで下がればそれ以上下げる必要はない。
	bool ear_damage_max_mode = true; 
	bool ear_damage_time_max_mode = false;
	double loudness_energy = 0;
	bakuage::AlignedPodVector<float> ear_damage_energies(band_count);
	bakuage::AlignedPodVector<float> noise_ear_damage_energies(band_count);
	for (int i = 0; i < count; i++) {
		for (int j = 0; j < band_count; j++) {
			double total_energy = (mid_mel_bands[band_count * i + j] + side_mel_bands[band_count * i + j]) * speaker_weights[j];
			loudness_energy += total_energy * loudness_weights[j];
			if (ear_damage_max_mode && ear_damage_time_max_mode) {
				ear_damage_energies[j] = std::max<double>(ear_damage_energies[j], total_energy * ear_damage_weights[j]);
				noise_ear_damage_energies[j] = std::max<double>(noise_ear_damage_energies[j], noise_mel_bands[j] * ear_damage_weights[j]);
			}
			else {
				ear_damage_energies[j] += total_energy * ear_damage_weights[j];
				noise_ear_damage_energies[j] += noise_mel_bands[j] * ear_damage_weights[j];
			}
		}
	}
	if (verbose2) {
		std::cerr << "CalculateMastering3Score noise_ear_damage_energies" << std::endl;
		for (int j = 0; j < band_count; j++) {
			std::cerr << noise_mel_bands[j] << " " << noise_ear_damage_energies[j] << std::endl;
		}
	}
	*loudness = 10 * std::log10(1e-10 + loudness_energy / (1e-37 + count));
	if (verbose2) std::cerr << "CalculateMastering3Score loudness " << *loudness << std::endl;
	double noise_ear_damage;
	if (ear_damage_max_mode) {
		*ear_damage = 10 * std::log10(1e-10 + *std::max_element(ear_damage_energies.begin(), ear_damage_energies.end()));
		noise_ear_damage = 10 * std::log10(1e-10 + *std::max_element(noise_ear_damage_energies.begin(), noise_ear_damage_energies.end()));
	}
	else {
		/**ear_damage = -1e100;
		for (int j = 0; j < band_count; j++) {
		*ear_damage = std::max<float>(*ear_damage, 10 * std::log10(1e-10 + ear_damage_energies[j] / (1e-37 + count)));
		}*/
		*ear_damage = 10 * std::log10(1e-10 + std::accumulate(ear_damage_energies.begin(), ear_damage_energies.end(), 0.0) / (1e-37 + count));
		noise_ear_damage = 10 * std::log10(1e-10 + std::accumulate(noise_ear_damage_energies.begin(), noise_ear_damage_energies.end(), 0.0) / (1e-37 + count));
	}
	if (verbose2) std::cerr << "CalculateMastering3Score ear_damage " << *ear_damage << std::endl;
	if (verbose2) std::cerr << "CalculateMastering3Score noise_ear_damage " << noise_ear_damage << std::endl;
	
	// ノイズの基準ラウドネス
	double noise_base_energy = 0;
	for (int j = 0; j < band_count; j++) {
		noise_base_energy += noise_mel_bands[j] * loudness_weights[j];
	}
	if (verbose2) std::cerr << "CalculateMastering3Score noise_base_energy " << noise_base_energy << std::endl;

	// ノイズ下MFCCの統計 + eigen用の統計
	double sn_energy_ratio = std::pow(10, -target_sn_db / 10.0);
	double noise_gain = sn_energy_ratio * std::pow(10, *loudness / 10.0) / noise_base_energy;
	// 耳へのダメージを一定とする方法はロバストでない。acoustic entropyの変化が緩慢な場合に、耳へのダメージ増加が軽視されてしまうから。
	// double noise_gain = sn_energy_ratio * std::pow(10, *ear_damage / 10.0) / noise_base_energy; // 耳へのダメージを一定とする
	// double noise_gain = sn_energy_ratio * std::pow(10, *ear_damage / 10.0) / std::pow(10, 0.1 * noise_ear_damage); // 耳へのダメージを一定とする
	if (verbose2) std::cerr << "CalculateMastering3Score noise_gain " << noise_gain << std::endl;
	std::vector<float> saturated_mid_log_mel_bands(band_count);
	std::vector<float> saturated_side_log_mel_bands(band_count);
	std::vector<float> saturated_concat_log_mel_bands(2 * band_count);
    std::vector<float> saturated_concat_log_mel_bands_prev(2 * band_count);
	std::vector<float> saturated_mid_mfcc(band_count);
	std::vector<float> saturated_side_mfcc(band_count);
	bakuage::Dct dct(band_count);
	std::vector<bakuage::Statistics> mid_mfcc_stats(band_count);
	std::vector<bakuage::Statistics> side_mfcc_stats(band_count);
	VectorStatistics bands_stats(2 * band_count);
    VectorStatistics diff_bands_stats(2 * band_count);
	Eigen::MatrixXd mel_bands_mat(2 * band_count, count);
	Eigen::MatrixXd log_mel_bands_mat(2 * band_count, count);

	std::vector<double> masking_table(band_count);
	for (int i = 0; i < band_count; i++) {
		masking_table[i] = std::pow(10, -12.0 * i * 0.1);
	}

	for (int i = 0; i < count; i++) {
		for (int j = 0; j < band_count; j++) {
			double noise_energy = 0;
			double mid_energy = 0;
			double side_energy = 0;
			for (int k = 0; k < band_count; k++) {
				const double weight = masking_table[std::abs(j - k)];
				noise_energy += noise_gain * noise_mel_bands[k] * loudness_weights[k] * weight;
				mid_energy += mid_mel_bands[band_count * i + k] * loudness_weights[k] * speaker_weights[k] * acoustic_entropy_mid_weights[k] * weight;
				side_energy += side_mel_bands[band_count * i + k] * loudness_weights[k] * speaker_weights[k] * acoustic_entropy_side_weights[k] * weight;
			}

			// maxではなくnoise_energy +のほうが、微分がゼロになりづらいので良い
			saturated_mid_log_mel_bands[j] = 10 * std::log10(noise_energy + mid_energy);
			saturated_side_log_mel_bands[j] = 10 * std::log10(noise_energy + side_energy);

			/*if (verbose2) {
			std::cerr << "CalculateMastering3Score noise_energy " << noise_energy
			<< " " << mid_energy << " " << side_energy << std::endl;
			}*/

			mel_bands_mat(j, i) = noise_energy + mid_energy;
			mel_bands_mat(band_count + j, i) = noise_energy + side_energy;
			log_mel_bands_mat(j, i) = 10 * std::log10(noise_energy + mid_energy);
			log_mel_bands_mat(band_count + j, i) = 10 * std::log10(noise_energy + side_energy);
		}

		// eigen
		for (int j = 0; j < band_count; j++) {
			saturated_concat_log_mel_bands[j] = saturated_mid_log_mel_bands[j];
			saturated_concat_log_mel_bands[band_count + j] = saturated_side_log_mel_bands[j];
		}
		bands_stats.Add(saturated_concat_log_mel_bands.data());
        if (i > 0) {
            // saturated_concat_log_mel_bands_prevを一時領域として利用
            for (int j = 0; j < 2 * band_count; j++) {
                saturated_concat_log_mel_bands_prev[j] = saturated_concat_log_mel_bands[j] - saturated_concat_log_mel_bands_prev[j];
            }
            diff_bands_stats.Add(saturated_concat_log_mel_bands_prev.data());
        }
        std::copy(saturated_concat_log_mel_bands.begin(), saturated_concat_log_mel_bands.end(), saturated_concat_log_mel_bands_prev.begin());

		// mfcc
		dct.DctType2(saturated_mid_log_mel_bands.data(), saturated_mid_mfcc.data());
		dct.DctType2(saturated_side_log_mel_bands.data(), saturated_side_mfcc.data());
		// 補正 (直交化)
		saturated_mid_mfcc[0] *= std::sqrt(0.5);
		saturated_side_mfcc[0] *= std::sqrt(0.5);
		for (int j = 0; j < band_count; j++) {
			//補正 (エネルギー保存)
			saturated_mid_mfcc[j] *= 2.0 / std::sqrt(band_count);
			saturated_side_mfcc[j] *= 2.0 / std::sqrt(band_count);

			// 以下のやりかたは妥当でない。理由は、MとSはゲインが増えると同じだけ動くから。
			// それを相殺するために、M - S (ステレオ度合いを表す) を用意するのが正解。
			// ステレオの場合だいたいM = Sになる。つまり45度
			// モノラルの場合、S = 0になる。つまり0度
			// これらの中間に向ける。
			/*double theta = (45.0 / 2) * M_PI / 180;
			double mid_plus_side = saturated_mid_mfcc[j] * std::cos(theta) + saturated_side_mfcc[j] * std::sin(theta);
			double mid_minus_side = -saturated_mid_mfcc[j] * std::sin(theta) + saturated_side_mfcc[j] * std::cos(theta);*/

			double sqrt_0_5 = std::sqrt(0.5);
			double mid_plus_side = sqrt_0_5 * (saturated_mid_mfcc[j] + saturated_side_mfcc[j]);
			double mid_minus_side = sqrt_0_5  * (saturated_mid_mfcc[j] - saturated_side_mfcc[j]);

			mid_mfcc_stats[j].Add(mid_plus_side);
			side_mfcc_stats[j].Add(mid_minus_side);
			//mid_mfcc_stats[j].Add(saturated_mid_mfcc[j]);
			//side_mfcc_stats[j].Add(saturated_side_mfcc[j]);
		}
	}

	// Acoustic Entropyについて
	// いろいろな考え方ができる。
	// 定性的に言えば、人が感じる情報量をあらわしたもの
	// 言い換えれば、圧縮したときのサイズ
	// 圧縮の方法はいろいろある。SVD、NMF、Vector Quantization, Sparse coding

	// Acoustic Entropy Eigen
	const std::string acoustic_entropy_mode = "svd2";
	if (acoustic_entropy_mode == "nmf") {
		Eigen::MatrixXd w, h;
		const int k = 8;
		bakuage::Nmf(mel_bands_mat, k, 1000, &w, &h);
		std::vector<bakuage::Statistics> stats(k);
		for (int i = 0; i < count; i++) {
			for (int j = 0; j < k; j++) {
				stats[j].Add(10 * std::log10(1 + bakuage::Sqr(h(j, i))));
			}
		}
		if (verbose2) {
			for (int i = 0; i < k; i++) {
				std::cerr << "CalculateMastering3Score nmf w ";
				for (int j = 0; j < k; j++) {
					std::cerr << (int)(w(j, i) * 9);
				}
				std::cerr << std::endl;
			}
			for (int j = 0; j < k; j++) {
				std::cerr << "CalculateMastering3Score nmf energy " << stats[j].mean() << " " << stats[j].variance() << std::endl;
			}
		}
		*acoustic_entropy_eigen = 0;
		for (int j = 0; j < k; j++) {
			*acoustic_entropy_eigen += 0.5 * std::log2(1 + stats[j].variance());
		}
	}
	else if (acoustic_entropy_mode == "svd2") {
		// 固有ベクトルのmaxを固有値にかけたバージョン。評価関数が変化しやすくなったけど、複雑
		// メリットは、帯域制限されても評価関数が変わらないこと
		// ノイズ音源の回転に対してロバストではない。(けど、まあそれも込みでありといえばあり)
		// 参考: grep 一意性 in http://elsur.jpn.org/reading_notes/Greenacre1984.pdf
		// スパース度合いをかけているともいえる (* L∞ norm / L2 norm)
		// それならそもそも分解のときにスパースになるような分解を選べば良い？
		// なぜかベースが大きくなりやすい。
        for (int k = 0; k < 2; k++) {
            Eigen::MatrixXd m = k == 0 ? bands_stats.covariance_as_matrix() : diff_bands_stats.covariance_as_matrix();
            // https://eigen.tuxfamily.org/dox/classEigen_1_1JacobiSVD.html
            // Singular values are always sorted in decreasing order.
            Eigen::BDCSVD<Eigen::MatrixXd> svd_solver(m, Eigen::ComputeFullU);

            std::vector<double> vec(2 * band_count);
            for (int i = 0; i < 2 * band_count; i++) {
                double m = 0;
                for (int j = 0; j < 2 * band_count; j++) {
                    m = std::max<double>(m, bakuage::Sqr(svd_solver.matrixU()(j, i)));
                }
                vec[i] = svd_solver.singularValues()(i, 0) * m;
            }

            bakuage::Statistics eigen_value_stats;
            eigen_value_stats.AddRange(vec.begin() + band_count / 2, vec.end());
            auto *target = k == 0 ? acoustic_entropy_eigen : diff_acoustic_entropy_eigen;
            *target = 0;
            // 固有ベクトルのmaxをかけているので、1dB^2がちょうどthresholdになる。
            auto eigen_threshold = 1;// eigen_value_stats.mean();
            if (verbose2) {
                std::cerr << "CalculateMastering3Score eigen threshold " << eigen_threshold << std::endl;
                for (int i = 0; i < 2 * band_count; i++) {
                    std::cerr << "CalculateMastering3Score vec " << vec[i] << std::endl;
                }
            }
            for (int j = 0; j < 2 * band_count; j++) {
                // *target += 0.5 * std::log2(1 + vec[j]);
                *target += 0.5 * std::log2(1 + SoftShrink(vec[j], eigen_threshold));
            }
        }
	}
	else if (acoustic_entropy_mode == "svd") {
        for (int k = 0; k < 2; k++) {
            auto eigen_values = k == 0 ? bands_stats.eigen_values() : diff_bands_stats.eigen_values();
            bakuage::Statistics eigen_value_stats;
            eigen_value_stats.AddRange(eigen_values.begin() + band_count / 2, eigen_values.end());
            auto *target = k == 0 ? acoustic_entropy_eigen : diff_acoustic_entropy_eigen;
            *target = 0;
            auto eigen_threshold = eigen_value_stats.mean();
            if (verbose2) std::cerr << "CalculateMastering3Score eigen threshold " << eigen_threshold << std::endl;
            for (int j = 0; j < 2 * band_count; j++) {
                //*target += SoftShrink(eigen_values[j], eigen_threshold);
                *target += 0.5 * std::log2(1 + SoftShrink(eigen_values[j], eigen_threshold));
                // *target += 0.5 * std::log2(1 + eigen_values[j]);
            }
            //*target = std::sqrt(*acoustic_entropy_eigen);
        }
	}
	
	// Acoustic Entropy MFCC calculate threshold for shrinkage
	bakuage::Statistics mid_vari_stats;
	bakuage::Statistics side_vari_stats;
	// for (int j = 0; j < band_count; j++) { // こっちだとエントロピーの差が出なさ過ぎる
	for (int j = band_count / 2; j < band_count; j++) {
		mid_vari_stats.Add(mid_mfcc_stats[j].variance());
		side_vari_stats.Add(side_mfcc_stats[j].variance());
	}
	double vari_threshold = std::max<double>(mid_vari_stats.mean(), side_vari_stats.mean());
	if (verbose2) std::cerr << "CalculateMastering3Score vari_threshold " << vari_threshold << std::endl;

	if (verbose2) {
		std::cerr << "CalculateMastering3Score variances" << std::endl;
		for (int j = 0; j < band_count; j++) {
			std::cerr << mid_mfcc_stats[j].variance() << " " << side_mfcc_stats[j].variance() << std::endl;
		}
	}
	// Acoustic Entropy MFCC
	*acoustic_entropy_mfcc = 0;
	for (int j = 0; j < band_count; j++) {
		// maxではなく1 + のほうが微分がゼロになりづらいので良い
		//*noise_acoustic_entropy += std::log2(1 + Shrink(mid_mfcc_stats[j].stddev(), vari_threshold));
		//*noise_acoustic_entropy += std::log2(1 + Shrink(side_mfcc_stats[j].stddev(), vari_threshold));

		// SoftShrinkのほうが微分ができるし、あと、引き算だからノイズの分を差し引ける。
		// 引き算はエネルギー空間で行う。spectrum subtraction methodに習って
		// http://www.wolframalpha.com/input/?i=ln(1+%2B+x)
		// http://www.wolframalpha.com/input/?i=ln(1+%2B+sqrt(x))
		//*noise_acoustic_entropy += std::log2(1 + std::sqrt(SoftShrink(mid_mfcc_stats[j].variance(), vari_threshold)));
		//*noise_acoustic_entropy += std::log2(1 + std::sqrt(SoftShrink(side_mfcc_stats[j].variance(), vari_threshold)));
		// 平方根だと最適化しづらい評価関数の形になると思う。
		*acoustic_entropy_mfcc += 0.5 * std::log2(1 + SoftShrink(mid_mfcc_stats[j].variance(), vari_threshold));
		*acoustic_entropy_mfcc += 0.5 * std::log2(1 + SoftShrink(side_mfcc_stats[j].variance(), vari_threshold));

		// ノイズレベルが上がったせいで、情報量が増えるのを防ぐために、引き算
		//*noise_acoustic_entropy += std::log2(1 + std::max<double>(0, mid_mfcc_stats[j].stddev() - std::sqrt(vari_threshold)));
		//*noise_acoustic_entropy += std::log2(1 + std::max<double>(0, side_mfcc_stats[j].stddev() - std::sqrt(vari_threshold)));

		// エネルギーの総和でやってみたバージョン
		//*noise_acoustic_entropy += std::max<double>(0, mid_mfcc_stats[j].variance() - vari_threshold);
		//*noise_acoustic_entropy += std::max<double>(0, side_mfcc_stats[j].variance() - vari_threshold);

		//*noise_acoustic_entropy += SoftShrink(mid_mfcc_stats[j].variance(), vari_threshold);
		//*noise_acoustic_entropy += SoftShrink(side_mfcc_stats[j].variance(), vari_threshold);
	}
	//*noise_acoustic_entropy = 10 * std::log2(1 + *noise_acoustic_entropy);
	if (verbose2) std::cerr << "CalculateMastering3Score acoustic_entropy_mfcc " << *acoustic_entropy_mfcc << std::endl;
	if (verbose2) std::cerr << "CalculateMastering3Score acoustic_entropy_eigen " << *acoustic_entropy_eigen << std::endl;
    if (verbose2) std::cerr << "CalculateMastering3Score diff_acoustic_entropy_eigen " << *diff_acoustic_entropy_eigen << std::endl;
}
}

#endif
