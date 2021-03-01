#include "bakuage/biquad_iir_filter.h"

#include "boost/math/tools/polynomial.hpp"
#include "bakuage/utils.h"

namespace bakuage {
    // z^-1, s表現どちらもありうる
    // b0 + b1 * s + b2 * s^2
    // ----------------------
    // 1 + a1 * s + a2 * s^2
    
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::Zero() {
        BiquadIIRCoef result;
        result.a1 = 0;
        result.a2 = 0;
        result.b0 = 0;
        result.b1 = 0;
        result.b2 = 0;
        return result;
    };
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::Zero();
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::Zero();
    
    // http://www.musicdsp.org/files/Audio-EQ-Cookbook.txt
    // z表現なので注意
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::CreateLowpass(Float normalized_freq, Float s) {
        Float w0 = 2 * M_PI * normalized_freq;
        // Float alpha = std::sin(w0) / (2 * q);
        Float alpha = 0.5 * std::sin(w0) * std::sqrt((2) * (1 / s - 1) + 2);
        Float c = std::cos(w0);
        BiquadIIRCoef result;
        Float inv_a0 = 1 / (1 + alpha);
        result.a1 = -2 * c;
        result.a2 = 1 - alpha;
        result.b0 = 0.5 * (1 - c);
        result.b1 = 1 - c;
        result.b2 = result.b0;
        return result * inv_a0;
    }
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::CreateLowpass(float, float);
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::CreateLowpass(double, double);
    
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::CreateButterworthLowpass(Float normalized_freq) {
        return CreateLowpass(normalized_freq, 1);
    }
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::CreateButterworthLowpass(float);
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::CreateButterworthLowpass(double);
    
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::operator *(Float other) const {
        BiquadIIRCoef result;
        result.a1 = a1 * other;
        result.a2 = a2 * other;
        result.b0 = b0 * other;
        result.b1 = b1 * other;
        result.b2 = b2 * other;
        return result;
    }
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::operator *(float) const;
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::operator *(double) const;
    
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::ChangeSampleFreq(int from, int to) const {
        BiquadIIRCoef s = BilinearInverseTransform(from);
        return s.BilinearTransform(to);
    }
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::ChangeSampleFreq(int, int) const;
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::ChangeSampleFreq(int, int) const;
    
    // s to z
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::BilinearTransform(int sample_freq) const {
        BiquadIIRCoef result;
        
        typedef boost::math::tools::polynomial<Float> Poly;
        
        static const Float z_data[2] = {0, 1};
        Poly z(z_data, 1); // z^-1
        
        // s = u / v
        Poly u = (2 * sample_freq) * (1 - z); // (2 / T) * (1 - z^-1)
        Poly v = 1 + z; // (1 + z^-1)
        
        Poly numerator = b0 * v * v + b1 * u * v + b2 * u * u;
        Poly denominator = v * v + a1 * u * v + a2 * u * u;
        
        // a0 == 1にする
        Float r = 1 / denominator[0];
        numerator *= r;
        denominator *= r;
        
        result.a1 = denominator[1];
        result.a2 = denominator[2];
        result.b0 = numerator[0];
        result.b1 = numerator[1];
        result.b2 = numerator[2];
        
        return result;
    }
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::BilinearTransform(int) const;
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::BilinearTransform(int) const;
    
    // z to s
    template <class Float>
    BiquadIIRCoef<Float> BiquadIIRCoef<Float>::BilinearInverseTransform(int sample_freq) const {
        BiquadIIRCoef result;
        
        typedef boost::math::tools::polynomial<Float> Poly;
        
        static const Float s_data[2] = {0, 1};
        Poly s(s_data, 1); // s
        
        // z-1 = u / v
        Poly u = 1 - s * (1.0 / (2 * sample_freq));
        Poly v = 1 + s * (1.0 / (2 * sample_freq));
        
        Poly numerator = b0 * v * v + b1 * u * v + b2 * u * u;
        Poly denominator = v * v + a1 * u * v + a2 * u * u;
        
        // a0 == 1にする
        Float r = 1 / denominator[0];
        numerator *= r;
        denominator *= r;
        
        result.a1 = denominator[1];
        result.a2 = denominator[2];
        result.b0 = numerator[0];
        result.b1 = numerator[1];
        result.b2 = numerator[2];
        
        return result;
    }
    template BiquadIIRCoef<float> BiquadIIRCoef<float>::BilinearInverseTransform(int) const;
    template BiquadIIRCoef<double> BiquadIIRCoef<double>::BilinearInverseTransform(int) const;
}
