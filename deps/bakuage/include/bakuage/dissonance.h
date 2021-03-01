#ifndef BAKUAGE_BAKUAGE_DISSONANCE_H_
#define BAKUAGE_BAKUAGE_DISSONANCE_H_

namespace bakuage {
    // reference
    // https://pypi.org/project/dissonant/
    // https://github.com/bzamecnik/dissonant/blob/master/dissonant/tuning.py (Hz)
    // https://essentia.upf.edu/documentation/reference/streaming_Dissonance.html
    // amp^2 = energy
    double DissonancePairSethares1993(double hz1, double hz2, double amp1, double amp2);
    
    // mfccはenergy sum modeを想定している
    template <class Float>
    void CalculateDissonance(Float *input, int channels, int samples, int sample_freq, Float *dissonance, bool tbb_parallel = false);
}

#endif /* BAKUAGE_BAKUAGE_DISSONANCE_H_ */
