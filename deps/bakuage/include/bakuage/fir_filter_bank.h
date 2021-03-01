#ifndef BAKUAGE_BAKUAGE_FIR_FILTER_BANK_H_
#define BAKUAGE_BAKUAGE_FIR_FILTER_BANK_H_

#include <algorithm>
#include <complex>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cassert>
#include "bakuage/dft.h"
#include "bakuage/memory.h"
#include "bakuage/utils.h"
#include "bakuage/vector_math.h"

namespace bakuage {
    // 複数のFIRをoverlap addで高速にたたみこみ
    // decimationもやってくれるフィルタ
    // Float must be float / double
    // コンストラクタだけではなく、AddFirでもmallocが発生するので、
    // VSTで使うときは注意
    // 全てのdecimationで割り切れる単位で入力する必要あり
    // FIRは複素数
    // 分割と合成両方できる
    // 今のところdecimationは2のるいじょうのみ
    // analysisの入力とsynthesisの出力は実数 (拡張したいなら、Complex版のClockを実装すれば良い)
    
    // 思考メモ
    // ゼロのところをはしょることで早くなるが、エイリアシングがでる
    // 元に戻すには？
    // 非ゼロのbin数がlen/dicimation以下である必要がある
    // かんたんにやるなら連続領域を指定した方が良いな
    // 折り返しが簡単な複素数でやったほうがかんたん
    
    template <typename Float = double>
    class FirFilterBank {
    public:
        struct BandConfig {
            int decimation;
            Float nonzero_base_normalized_freq;
            AlignedPodVector<std::complex<Float>> analysis_fir;
            AlignedPodVector<std::complex<Float>> synthesis_fir;
        };
        
        struct Config {
            std::vector<BandConfig> bands;
        };
        
        FirFilterBank(const Config &config):
        max_fir_size_(0),
        max_decimation_(0),
        analysis_output_buffer_pos_(0),
        synthesis_output_buffer_pos_(0)
        {
            for (const auto &band_config: config.bands) {
                max_fir_size_ = std::max<int>(max_fir_size_, band_config.analysis_fir.size());
                max_fir_size_ = std::max<int>(max_fir_size_, band_config.synthesis_fir.size());
                max_decimation_ = std::max<int>(max_decimation_, band_config.decimation);
                assert(CeilPowerOf2(band_config.decimation) == band_config.decimation);
            }
            assert(max_fir_size_ > 0);
            assert(max_decimation_ > 0);
            extended_size_ = CeilPowerOf2(2 * std::max<int>(max_fir_size_, max_decimation_));
            assert(extended_size_ >= 2);
            assert(extended_size_ >= max_decimation_);
            work_.resize(extended_size_);
            work2_.resize(extended_size_);
            extended_real_spec_size_ = extended_size_ / 2 + 1;
            work_spec_.resize(extended_size_); // 実数FFT用だが、一時的に負の周波数側まで使う
            work_spec2_.resize(extended_size_);
            
            max_process_len_ = extended_size_ - CeilInt(max_fir_size_ - 1, max_decimation_);
            assert(max_process_len_ >= max_decimation_);
            
            real_dft_ = std::unique_ptr<RealDft<Float>>(new RealDft<Float>(extended_size_));
            dft_.emplace(extended_size_, extended_size_);
            
            for (const auto &band_config: config.bands) {
                Band band;
                band.decimation = band_config.decimation;
                
                // nonzero_base_binを0から1にする
                const Float freq = band_config.nonzero_base_normalized_freq - std::floor(band_config.nonzero_base_normalized_freq);
                band.nonzero_base_bin =  std::floor(freq * extended_size_);
                
                const int decimated_len = extended_size_ / band.decimation;
                if (dft_.find(decimated_len) == dft_.end()) {
                    dft_.emplace(decimated_len, decimated_len);
                }
                
                // calc fir spec decimated
                for (int idx = 0; idx < 2; idx++) {
                    const auto &fir = idx == 0 ? band_config.analysis_fir : band_config.synthesis_fir;
                    auto &shifted_fir_spec = idx == 0 ? band.analysis_shifted_fir_spec : band.synthesis_shifted_fir_spec;
                    
                    TypedFillZero(work2_.data(), work2_.size());
                    for (int i = 0; i < fir.size(); i++) {
                        work2_[i] = fir[i];
                    }
                    dft_.at(extended_size_).Forward((Float *)work2_.data(), (Float *)work_spec2_.data());
                    
                    shifted_fir_spec.resize(decimated_len);
					// アップサンプリングしたときにゲインを戻す
                    const Float normalization_scale = 1.0 / std::sqrt(extended_size_ * extended_size_) * (idx == 0 ? 1 : band.decimation);
                    for (int i = 0; i < decimated_len; i++) {
                        int bin = (band.nonzero_base_bin + i) % extended_size_;
                        shifted_fir_spec[i] = work_spec2_[bin] * normalization_scale;
                    }
                }
                
                bands_.emplace_back(std::move(band));
                const int decimated_output_buffer_len = output_buffer_size() / band.decimation;
                assert(decimated_output_buffer_len > 0);
                analysis_output_buffer_.emplace_back(decimated_output_buffer_len);
            }
            synthesis_output_buffer_.resize(output_buffer_size());
        }
        
        // lenはdecimation前で指定
        void SynthesisClock(const std::complex<Float> **input, int len, Float *output) {
            // 元に戻すには？
            // アップサンプルして同じFIRをかけて混ぜれば良い
            
            int input_pos = 0;
            while (input_pos < len) {
                const int remaining_len = len - input_pos;
                const int process_len = std::min<int>(remaining_len, max_process_len_);
                const int convoluted_size = process_len + max_fir_size_ - 1;
                
                // shift output buffer if needed
                if (synthesis_output_buffer_pos_ + convoluted_size > output_buffer_size()) {
                    const int remaining_size = output_buffer_size() - synthesis_output_buffer_pos_;
                    TypedMemmove(synthesis_output_buffer_.data(), synthesis_output_buffer_.data() + synthesis_output_buffer_pos_, remaining_size);
                    TypedFillZero(synthesis_output_buffer_.data() + remaining_size, synthesis_output_buffer_pos_);
                    synthesis_output_buffer_pos_ = 0;
                }
                
                // work_spec_にcomposedされたspecを作る (一時的に負の周波数側まで使う)
                TypedFillZero(work_spec_.data(), extended_size_);
                
                // band compose
                for (int i = 0; i < bands_.size(); i++) {
                    const auto &band = bands_[i];
                    const int decimated_len = extended_size_ / band.decimation;
                    
                    // FFT to calc decimated wave
                    const int decimated_input_pos = input_pos / band.decimation;
                    const int decimated_process_len = process_len / band.decimation;
                    TypedMemcpy(work2_.data(), input[i] + decimated_input_pos, decimated_process_len);
                    TypedFillZero(work2_.data() + decimated_process_len, decimated_len - decimated_process_len);
                    dft_.at(decimated_len).Forward((Float *)work2_.data(), (Float *)work_spec2_.data());
                    
                    // calc convoluted and interpolated spec
                    int bin = band.nonzero_base_bin;
                    int decimated_bin = bin % decimated_len;
#if 1
                    {
                        int j = 0;
                        while (j < decimated_len) {
                            int step_size = std::min<int>(decimated_len - j, std::min<int>(extended_size_ - bin, decimated_len - decimated_bin));
                            
                            VectorMadInplace(work_spec2_.data() + decimated_bin, band.synthesis_shifted_fir_spec.data() + j, work_spec_.data() + bin, step_size);
                            
                            bin += step_size;
                            decimated_bin += step_size;
                            j += step_size;
                            if (bin >= extended_size_) bin = 0;
                            if (decimated_bin >= decimated_len) decimated_bin = 0;
                        }
                    }
#else
                    // 元のコード (最適化版が複雑なのでメモとして残しておく)
                    for (int j = 0; j < decimated_len; j++) {
                        if (bin >= extended_real_spec_size_) {
                            work_spec_[2 * (extended_real_spec_size_ - 1) - bin] += std::conj(work_spec2_[decimated_bin] * band.synthesis_shifted_fir_spec[j]);
                        } else {
                            work_spec_[bin] += work_spec2_[decimated_bin] * band.synthesis_shifted_fir_spec[j];
                        }
                        bin++;
                        decimated_bin++;
                        if (bin >= extended_size_) bin = 0;
                        if (decimated_bin >= decimated_len) decimated_bin = 0;
                    }
#endif
                }
                
                // 負の周波数側を正側に足しこむ
                VectorReverseInplace(&work_spec_[extended_size_ / 2 + 1], extended_size_ / 2 - 1);
                VectorConjInplace(&work_spec_[extended_size_ / 2 + 1], extended_size_ / 2 - 1);
                VectorAddInplace(&work_spec_[extended_size_ / 2 + 1], &work_spec_[1], extended_size_ / 2 - 1);
                
                work_spec_[0].imag(0);
                VectorMulConstantInplace<std::complex<Float>>(0.5, work_spec_.data() + 1, extended_size_ / 2 - 1);
                work_spec_[extended_size_ / 2].imag(0);
                
                // real ifft to calc convoluted and composed and upsampled wav
                real_dft_->Backward((Float *)work_spec_.data(), work_.data());
                
                // add and output
                VectorAdd(work_.data(), synthesis_output_buffer_.data() + synthesis_output_buffer_pos_, output, process_len);
                
                // add to buffer
                VectorAddInplace(work_.data() + process_len, synthesis_output_buffer_.data() + synthesis_output_buffer_pos_ + process_len, convoluted_size - process_len);
                
                synthesis_output_buffer_pos_ += process_len;
                input_pos += process_len;
                output += process_len;
            }
        }
        
        // 長さは全てのdecimationで割り切れる必要がある
        void AnalysisClock(const Float *bg, const Float *ed, std::complex<Float> **output) {
            assert((ed - bg) % max_decimation_ == 0);
            
            int output_pos = 0;
            while (bg < ed) {
                const int remaining_len = ed - bg;
                const int process_len = std::min<int>(remaining_len, max_process_len_);
                const int convoluted_size = process_len + max_fir_size_ - 1;
                
                // shift if needed
                if (analysis_output_buffer_pos_ + convoluted_size > output_buffer_size()) {
                    for (int i = 0; i < bands_.size(); i++) {
                        const auto &band = bands_[i];
                        const int decimated_pos = analysis_output_buffer_pos_ / band.decimation;
                        const int remaining_size = analysis_output_buffer_[i].size() - decimated_pos;
                        TypedMemmove(analysis_output_buffer_[i].data(), analysis_output_buffer_[i].data() + decimated_pos, remaining_size);
                        TypedFillZero(analysis_output_buffer_[i].data() + remaining_size, decimated_pos);
                    }
                    analysis_output_buffer_pos_ = 0;
                }
                
                // real fft src
                TypedMemcpy(work_.data(), bg, process_len);
                TypedFillZero(work_.data() + process_len, extended_size_ - process_len);
                real_dft_->Forward(work_.data(), (Float *)work_spec_.data());
            
                // 負の周波数を作る (あまり重くない)
                TypedMemcpy(&work_spec_[extended_size_ / 2 + 1], &work_spec_[1], extended_size_ / 2 - 1);
                VectorReverseInplace(&work_spec_[extended_size_ / 2 + 1], extended_size_ / 2 - 1);
                VectorConjInplace(&work_spec_[extended_size_ / 2 + 1], extended_size_ / 2 - 1);
                
                // band split
                for (int i = 0; i < bands_.size(); i++) {
                    // calc convoluted and decimated spec
                    const auto &band = bands_[i];
                    const int decimated_len = extended_size_ / band.decimation;
                    int bin = band.nonzero_base_bin;
                    int decimated_bin = bin % decimated_len;
                    // ここが重い
#if 1
                    {
                        int j = 0;
                        while (j < decimated_len) {
                            int step_size = std::min<int>(decimated_len - j, std::min<int>(extended_size_ - bin, decimated_len - decimated_bin));
                            
                            VectorMul(work_spec_.data() + bin, band.analysis_shifted_fir_spec.data() + j, work_spec2_.data() + decimated_bin, step_size);
                            
                            bin += step_size;
                            decimated_bin += step_size;
                            j += step_size;
                            if (bin >= extended_size_) bin = 0;
                            if (decimated_bin >= decimated_len) decimated_bin = 0;
                        }
                    }
#else
                    // 元のコード (最適化版が複雑なのでメモとして残しておく)
                    for (int j = 0; j < decimated_len; j++) {
                        if (bin >= extended_real_spec_size_) {
                            work_spec2_[decimated_bin] = std::conj(work_spec_[2 * (extended_real_spec_size_ - 1) - bin]) * band.analysis_shifted_fir_spec[j];
                        } else {
                            work_spec2_[decimated_bin] = work_spec_[bin] * band.analysis_shifted_fir_spec[j];
                        }
                        bin++;
                        decimated_bin++;
                        if (bin >= extended_size_) bin = 0;
                        if (decimated_bin >= decimated_len) decimated_bin = 0;
                    }
#endif
                    
                    // IFFT to calc convoluted and decimated wave
                    dft_.at(decimated_len).Backward((Float *)work_spec2_.data(), (Float *)work2_.data());
                    
                    // add and output
                    const int decimated_pos = analysis_output_buffer_pos_ / band.decimation;
                    const int decimated_convoluted_size = convoluted_size / band.decimation;
                    const int decimated_process_len = process_len / band.decimation;
                    VectorAdd(work2_.data(), analysis_output_buffer_[i].data() + decimated_pos, output[i] + output_pos / band.decimation, decimated_process_len);
                    
                    // add to buffer
                    VectorAddInplace(work2_.data() + decimated_process_len, analysis_output_buffer_[i].data() + decimated_pos + decimated_process_len, decimated_convoluted_size - decimated_process_len);
                }
                
                analysis_output_buffer_pos_ += process_len;
                bg += process_len;
                output_pos += process_len;
            }
        };
    private:
        struct Band {
            // shifted_fir_specは[nonzero_base_bin, nonzero_base_bin + extended_len / decimation)に対応
            // それ以外は無視される(共役成分も無視)
            // shifted_fir_specは、各BandのFFT正規化と全体でのFFT正規化の、両方の補正を含む
            int decimation;
            int nonzero_base_bin;
            // analysis用のは帯域分割用で、ちゃんと再構成できるようにとなりのバンドと調整する必要があるが、
            // synthesis用のはエイリアシングノイズ除去用なので、analysis用よりも広帯域なら良い
            AlignedPodVector<std::complex<Float>> synthesis_shifted_fir_spec; // extended_len / decimation
            AlignedPodVector<std::complex<Float>> analysis_shifted_fir_spec; // extended_len / decimation
        };
        
        int output_buffer_size() const { return 2 * extended_size_; }
        int extended_size_;
        int extended_real_spec_size_;
            
        int max_fir_size_;
        int max_decimation_;
        int max_process_len_;
        std::vector<Band> bands_;
        
        // ライフタイム: 関数内
        AlignedPodVector<Float> work_; // extended_size_
        AlignedPodVector<std::complex<Float>> work2_; // extended_size_
        AlignedPodVector<std::complex<Float>> work_spec_; // extended_real_spec_size
        AlignedPodVector<std::complex<Float>> work_spec2_; // extended_size_
        
        // ライフタイム: ずっと
        std::vector<AlignedPodVector<std::complex<Float>>> analysis_output_buffer_; // 2 * extended_size_ / decimation
        int analysis_output_buffer_pos_; // decimation前の単位
        AlignedPodVector<Float> synthesis_output_buffer_; // 2 * extended_size_ / decimation
        int synthesis_output_buffer_pos_; // decimation前の単位
        
        std::unique_ptr<RealDft<Float>> real_dft_;
        std::unordered_map<int, Dft<Float>> dft_; // [len]
    };
}

#endif

