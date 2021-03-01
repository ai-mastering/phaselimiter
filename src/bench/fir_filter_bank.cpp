
#include <benchmark/benchmark.h>
#include "bakuage/fir_design.h"
#include "bakuage/fir_filter_bank.h"

void BM_FirFilterBankAnalysis(benchmark::State& state) {
    typedef float Float;
    using namespace bakuage;
    typedef FirFilterBank<Float> FilterBank;
    
    FilterBank::Config config;
    
    const int fir_size = 129;
    const float freqs[] = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    std::vector<std::complex<Float>> sum_fir(fir_size);
    for (int i = 0; i < 10; i++) {
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
    
    FilterBank filter_bank(config);
    
    const int process_size = 256;
    std::vector<Float> input(process_size);
    for (int j = 0; j < process_size; j++) {
        input[j] = j;
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
    
    // warmup
    filter_bank.AnalysisClock(input.data(), input.data() + process_size, output_ptr.data());
    
    for (auto _ : state) {
        filter_bank.AnalysisClock(input.data(), input.data() + process_size, output_ptr.data());
    }
}
BENCHMARK(BM_FirFilterBankAnalysis);

void BM_FirFilterBankSynthesis(benchmark::State& state) {
    typedef float Float;
    using namespace bakuage;
    typedef FirFilterBank<Float> FilterBank;
    
    FilterBank::Config config;
    
    const int fir_size = 129;
    const float freqs[] = { 0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 };
    std::vector<std::complex<Float>> sum_fir(fir_size);
    for (int i = 0; i < 10; i++) {
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
    
    FilterBank filter_bank(config);
    
    const int process_size = 256;
    std::vector<Float> input(process_size);
    for (int j = 0; j < process_size; j++) {
        input[j] = j;
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
    
    // warmup
    filter_bank.SynthesisClock(const_output_ptr.data(), process_size, input.data());
    
    for (auto _ : state) {
        filter_bank.SynthesisClock(const_output_ptr.data(), process_size, input.data());
    }
}
BENCHMARK(BM_FirFilterBankSynthesis);
