#include <iostream>
#include <vector>
#include <sstream>
#include <chrono>
#include <string>
#include <stdexcept>
#include <random>

#include "gtest/gtest.h"

#include "bakuage/utils.h"
#include "bakuage/delay_filter.h"
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter.h"
#include "bakuage/fir_filter_bank.h"

namespace {
    struct FirFilterBankTestParam {
        int process_size;

        int fir_size;
        int decimation;
        float bg_freq;
        float ed_freq;
        
        // 2は他のバンドの影響を受けないかの確認用。結果の確認には使わない
        int fir_size2;
        int decimation2;
        
        const char *name;
        
        friend std::ostream& operator<<(std::ostream& os, const FirFilterBankTestParam& param) {
            os << "FirFilterBankTestParam"
            << " name " << (param.name ? param.name : "no name")
            << " process_size " << param.process_size
            << " fir_size " << param.fir_size
            << " decimation " << param.decimation
            << " bg_freq " << param.bg_freq
            << " ed_freq " << param.ed_freq
            << " fir_size2 " << param.fir_size2
            << " decimation2 " << param.decimation2
            ;
            return os;
        }
    };
    const FirFilterBankTestParam test_params[] = {
#if 1
        { 1, 1, 1, 0, 0.5, 1, 1, "trivial" },
        { 2, 1, 1, 0, 0.5, 1, 1, "trivial + process size" },
        { 256, 1, 1, 0, 0.5, 1, 1, "trivial + large process size" },
        { 128, 3, 1, 0, 0.5, 1, 1, "trivial + fir size" },
        { 1, 65, 1, 0, 0.5, 1, 1, "trivial + large fir size" },
        { 1, 1, 1, 0.1, 0.2, 1, 1, "trivial + freq" },
        { 2, 1, 1, 0, 0.5, 1, 2, "trivial + band2 decimation" }, // ここでanalysis_output_bufferがおかしくなることがある (多分すでに修正済みのメモリ破壊)
        { 1, 1, 1, 0, 0.5, 3, 1, "trivial + band2 fir size"  },
        { 2, 1, 1, 0, 0.5, 3, 2, "trivial + band2 fir size + decimation"  },
        { 1, 65, 1, 0.0, 0.5, 1, 1, "large flat fir" },
        { 1, 65, 1, 0.0, 0.1, 1, 1, "large lowpass fir" },
        { 1, 65, 1, 0.4, 0.5, 1, 1, "large highpass fir" },
        { 1, 65, 1, 0.5, 1.0, 1, 1, "large hilbert fir" },
        { 1, 65, 1, 0.1, 0.2, 1, 1, "large band pass fir" },
        { 1, 65, 1, 0.1, 0.2, 1, 1, "large band pass fir 2" },
        { 256, 65, 1, 0.1, 0.2, 1, 1, "large band pass fir + large process size" },
        { 2, 65, 2, 0.1, 0.2, 1, 1, "large band pass fir + decimation" },
        { 4, 129, 4, 0.1, 0.2, 1, 1, "large band pass fir + decimation 4" },
#endif
    };

    class FirFilterBankTest : public ::testing::TestWithParam<FirFilterBankTestParam> {};
    
    template <class Float>
    std::vector<std::complex<Float>> HilbertTransform(const std::vector<std::complex<Float>> &input) {
        std::vector<std::complex<Float>> output(input);
        std::vector<std::complex<Float>> temp(input);
        const int len = input.size();
        bakuage::Dft<Float> dft(len);
        dft.Forward((Float *)input.data(), (Float *)temp.data());
        temp[0] *= 0.5;
        if (len % 2 == 0) {
            temp[len / 2] *= 0.5;
        }
        for (int i = len / 2 + 1; i < len; i++) temp[i] = 0;
        for (int i = 0; i < len; i++) temp[i] /= len;
        dft.Backward((Float *)temp.data(), (Float *)output.data());
        return output;
    }
}

// テスト補足
// filter設計
// keiser窓で窓関数法で設計している。decimationのテストもしているので、
// エイリアシングノイズが意図通りのレベルになっているかも確認できているはず。

// analysis
// ランダムな信号を入力し、通常のFirFilterの出力と比較するテスト
#if 1
TEST_P(FirFilterBankTest, Analysis) {
    typedef float Float;
    using namespace bakuage;
    typedef FirFilterBank<Float> FilterBank;
    const auto param = GetParam();
    
    FilterBank::Config config;
    FilterBank::BandConfig band;
    
    // http://www.mk.ecei.tohoku.ac.jp/jspmatlab/pdf/matdsp4.pdf
    const auto fir = CalculateBandPassFirComplex<Float>(param.bg_freq, param.ed_freq, param.fir_size, 7);
    
    // 一つ目のバンドを作る
    band.decimation = param.decimation;
    band.analysis_fir = AlignedPodVector<std::complex<Float>>(fir.begin(), fir.end());
    band.nonzero_base_normalized_freq = (param.bg_freq + param.ed_freq) / 2 - 0.5 / param.decimation; // 帯域がdecimate後の領域の真ん中にくるようにする
    config.bands.emplace_back(band);
    
    // 適当に二つ目のバンドを作る
    band.decimation = param.decimation2;
    band.analysis_fir.resize(param.fir_size2);
    for (int i = 0; i < param.fir_size2; i++) band.analysis_fir[i] = i;
    band.nonzero_base_normalized_freq = (param.bg_freq + param.ed_freq) / 2;
    config.bands.emplace_back(band);
    
    FilterBank filter_bank(config);
    FirFilter<std::complex<Float>> reference_filter(fir.begin(), fir.end());
    
    const int process_size = param.process_size;
    std::mt19937 engine(1);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i = 0; i < 100 * param.fir_size / process_size + 1; i++) {
        std::vector<Float> input(process_size);
        for (int j = 0; j < process_size; j++) {
            input[j] = dist(engine);
        }
        
        std::vector<std::complex<Float>> reference_output(process_size);
        for (int j = 0; j < process_size; j++) {
            reference_output[j] = reference_filter.Clock(input[j]);
        }
        
        std::vector<std::complex<Float>> output(process_size / param.decimation);
        std::vector<std::complex<Float>> output2(process_size / param.decimation2);
        std::complex<Float> *output_ptr[2];
        output_ptr[0] = output.data();
        output_ptr[1] = output2.data();
        filter_bank.AnalysisClock(input.data(), input.data() + process_size, output_ptr);
        
        for (int j = 0; j < process_size / param.decimation; j++) {
            EXPECT_LT(std::abs(reference_output[param.decimation * j] - output[j]), 1e-6);
        }
    }
}
#endif

// synthesis
// ランダムな信号を入力し、通常のFirFilterの出力と比較するテスト
#if 1
TEST_P(FirFilterBankTest, Synthesis) {
    typedef float Float;
    using namespace bakuage;
    typedef FirFilterBank<Float> FilterBank;
    const auto param = GetParam();
    
    FilterBank::Config config;
    FilterBank::BandConfig band;
    
    // http://www.mk.ecei.tohoku.ac.jp/jspmatlab/pdf/matdsp4.pdf
    const auto fir = CalculateBandPassFirComplex<Float>(param.bg_freq, param.ed_freq, param.fir_size, 7);
    
    // 一つ目のバンドを作る
    band.decimation = param.decimation;
#if 0
    band.fir = AlignedPodVector<std::complex<Float>>(fir_hilbert.begin(), fir_hilbert.end());
#else
    band.synthesis_fir = AlignedPodVector<std::complex<Float>>(fir.begin(), fir.end());
#endif
    band.nonzero_base_normalized_freq = (param.bg_freq + param.ed_freq) / 2 - 0.5 / param.decimation; // 帯域がdecimate後の領域の真ん中にくるようにする
    config.bands.emplace_back(band);
    
    // 適当に二つ目のバンドを作る
    band.decimation = param.decimation2;
    band.synthesis_fir.resize(param.fir_size2);
    for (int i = 0; i < param.fir_size2; i++) band.synthesis_fir[i] = i;
    band.nonzero_base_normalized_freq = (param.bg_freq + param.ed_freq) / 2;
    config.bands.emplace_back(band);
    
    FilterBank filter_bank(config);
#if 0
    FirFilter<std::complex<Float>> reference_filter(fir_hilbert.begin(), fir_hilbert.end());
#else
    FirFilter<std::complex<Float>> reference_filter(fir.begin(), fir.end());
#endif
    
    const int process_size = param.process_size;
    std::mt19937 engine(1);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i = 0; i < 100 * param.fir_size / process_size + 1; i++) {
        std::vector<std::complex<Float>> input(process_size);
        std::vector<std::complex<Float>> input2(process_size);
        for (int j = 0; j < process_size / param.decimation; j++) {
            input[j] = std::complex<Float>(dist(engine), dist(engine));
        }
        for (int j = 0; j < process_size / param.decimation2; j++) {
            input2[j] = 0; // std::complex<Float>(dist(engine), dist(engine));
        }
        
        std::vector<Float> reference_output(process_size);
        for (int j = 0; j < process_size; j++) {
            if (j % param.decimation == 0) {
                reference_output[j] = reference_filter.Clock(input[j / param.decimation]).real() * param.decimation;
            } else {
                reference_output[j] = reference_filter.Clock(std::complex<Float>(0.0f, 0.0f)).real() * param.decimation;
            }
        }
        
        std::vector<Float> output(process_size);
        const std::complex<Float> *input_ptr[2];
        input_ptr[0] = input.data();
        input_ptr[1] = input2.data();
        filter_bank.SynthesisClock(input_ptr, process_size, output.data());
        
        for (int j = 0; j < process_size / param.decimation; j++) {
            EXPECT_LT(std::abs(reference_output[j] - output[j]), 1e-6);
        }
    }
}
#endif

#if 1
// reconstruction test
// 完全再構成のフィルタバンクを作って
// ランダムな信号を与えて、analysis -> synthesisとやって、元に戻るか観察するテスト
// バンド境界でちゃんと戻るかがポイント
TEST(FirFilterBank, Reconstruction) {
    typedef float Float;
    using namespace bakuage;
    typedef FirFilterBank<Float> FilterBank;
    
    FilterBank::Config config;
    
    const int fir_size = 129;
    const int delay = 2 * ((fir_size - 1) / 2);
    const float freqs[] = { 0, 0.5, 1.0 };
    std::vector<std::complex<Float>> sum_fir(fir_size);
    for (int i = 0; i < 2; i++) {
        FilterBank::BandConfig band;
        
        // http://www.mk.ecei.tohoku.ac.jp/jspmatlab/pdf/matdsp4.pdf
        const auto fir = CalculateBandPassFirComplex<Float>(freqs[i], freqs[i + 1], fir_size, 7);
        const float band_width = freqs[i + 1] - freqs[i];
        const float center_freq = (freqs[i + 1] + freqs[i]) / 2;
        band.decimation = std::max<int>(1, std::floor(1.0 / (1.2 * band_width)));
        const float min_freq = center_freq - 0.5 / band.decimation;
        const float max_freq = center_freq + 0.5 / band.decimation;
        band.analysis_fir = AlignedPodVector<std::complex<Float>>(fir.begin(), fir.end());
        const auto synthesis_fir = CalculateBandPassFirComplex<Float>((freqs[i] + min_freq) / 2, (freqs[i + 1] + max_freq) / 2, fir_size, 7);
        band.synthesis_fir = AlignedPodVector<std::complex<Float>>(synthesis_fir.begin(), synthesis_fir.end());
        band.nonzero_base_normalized_freq = min_freq; // 帯域がdecimate後の領域の真ん中にくるようにする
        config.bands.emplace_back(band);
        
        for (int i = 0; i < fir_size; i++) {
            sum_fir[i] += fir[i];
        }
    }
    for (int i = 0; i < fir_size; i++) {
        if (i == fir_size / 2) {
            EXPECT_LT(std::abs(sum_fir[i] - std::complex<Float>(1, 0)), 1e-6);
        } else {
            EXPECT_LT(std::abs(sum_fir[i]), 1e-6);
        }
    }
    
    FilterBank filter_bank(config);
    DelayFilter<Float> delay_filter(delay);
    DelayFilter<Float> half_delay_filter(delay / 2);
    
    const int process_size = 256;
    std::mt19937 engine(1);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    for (int i = 0; i < 10 * fir_size / process_size + 1; i++) {
        std::vector<Float> input(process_size);
        for (int j = 0; j < process_size; j++) {
            input[j] = dist(engine);
        }
        
        std::vector<Float> reference_output(process_size);
        std::vector<Float> reference_output_half(process_size);
        for (int j = 0; j < process_size; j++) {
            reference_output[j] = delay_filter.Clock(input[j]);
            reference_output_half[j] = half_delay_filter.Clock(input[j]);
        }
        
        std::vector<std::vector<std::complex<Float>>> outputs;
        std::vector<std::complex<Float> *> output_ptr;
        std::vector<const std::complex<Float> *> const_output_ptr;
        for (int k = 0; k < config.bands.size(); k++) {
            outputs.emplace_back(process_size / config.bands[k].decimation);
        }
        for (int k = 0; k < config.bands.size(); k++) {
            output_ptr.emplace_back(outputs[k].data());
            const_output_ptr.emplace_back(outputs[k].data());
        }
        
        filter_bank.AnalysisClock(input.data(), input.data() + process_size, output_ptr.data());
        
#if 1
        std::vector<std::complex<Float>> sum_output(process_size);
        for (int k = 0; k < outputs.size(); k++) {
            for (int j = 0; j < process_size; j++) {
                sum_output[j] += outputs[k][j];
            }
        }
        for (int j = 0; j < process_size; j++) {
            EXPECT_LT(std::abs(reference_output_half[j] - sum_output[j]), 1e-6);
        }
#endif
    
        std::vector<Float> reconstruct(process_size);
        filter_bank.SynthesisClock(const_output_ptr.data(), process_size, reconstruct.data());
        
        for (int j = 0; j < process_size; j++) {
            EXPECT_LT(std::abs(reference_output[j] - reconstruct[j]), 1e-6);
        }
    }
}
#endif

INSTANTIATE_TEST_CASE_P(FirFilterBankTestInstance,
                        FirFilterBankTest,
                        ::testing::ValuesIn(test_params));
