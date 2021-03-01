#ifndef BAKUAGE_BAKUAGE_BIQUAD_IIR_FILTER_H_
#define BAKUAGE_BAKUAGE_BIQUAD_IIR_FILTER_H_

namespace bakuage {
	// z^-1, s表現どちらもありうる
    // b0 + b1 * s + b2 * s^2
    // ----------------------
    // 1 + a1 * s + a2 * s^2
    template <class Float = double>
    struct BiquadIIRCoef {    
        static BiquadIIRCoef Zero();

        // http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
        // z表現なので注意
        static BiquadIIRCoef CreateLowpass(Float normalized_freq, Float s);
        static BiquadIIRCoef CreateButterworthLowpass(Float normalized_freq);

        BiquadIIRCoef operator *(Float other) const;

        BiquadIIRCoef ChangeSampleFreq(int from, int to) const;

        // s to z
        BiquadIIRCoef BilinearTransform(int sample_freq) const;

        // z to s
        BiquadIIRCoef BilinearInverseTransform(int sample_freq) const;

        Float a1, a2, b0, b1, b2;
    };

    template <class Float = double>
    class BiquadIIRFilter {
    public:
        typedef BiquadIIRCoef<Float> Coef;

        BiquadIIRFilter(): coef_(Coef::Zero()) {
            ClearState();
        }

        BiquadIIRFilter(const Coef &_coef): coef_(_coef) {
            ClearState();
        }

        void set_coef(const Coef &_coef) { coef_ = _coef; }

        const Coef &coef() const { return coef_; }

        Float Clock(const Float &x) {
            // 足し算を並列実行
            const Float y = (coef_.b0 * x + coef_.b1 * x1 + coef_.b2 * x2)
                - (coef_.a1 * y1 + coef_.a2 * y2);

            x2 = x1;
            x1 = x;
            y2 = y1;
            y1 = y;

            return y;
        }

        void ClearState() {
            y1 = y2 = x1 = x2 = 0;
        }
    private:
        Float y1, y2, x1, x2;
        Coef coef_;
    };
}

#endif 
