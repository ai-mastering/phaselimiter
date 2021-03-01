#include "bakuage/vector_math.h"

#include <complex>
#include "ipp.h"
#include "bakuage/memory.h"

static_assert(sizeof(std::complex<float>) == sizeof(Ipp32fc), "ipp float complex size must be same as std complex size");
static_assert(sizeof(std::complex<double>) == sizeof(Ipp64fc), "ipp double complex size must be same as std complex size");

namespace bakuage {

    template <>
    void VectorMulConstantInplace<float, float>(const float &c, float *output, int n) {
        ippsMulC_32f_I(c, output, n);
    }

    template <>
    void VectorMulConstantInplace<double, float>(const double &c, float *output, int n) {
        VectorMulConstantInplace<float, float>(c, output, n);
    }

    template <>
    void VectorMulConstantInplace<double, double>(const double &c, double *output, int n) {
        ippsMulC_64f_I(c, output, n);
    }

    template <>
    void VectorMulConstantInplace<float, double>(const float &c, double *output, int n) {
        VectorMulConstantInplace<double, double>(c, output, n);
    }

    template <>
    void VectorMulConstantInplace<float, std::complex<float>>(const float &c, std::complex<float> *output, int n) {
        VectorMulConstantInplace(c, (float *)output, 2 * n);
    }

    template <>
    void VectorMulConstantInplace<double, std::complex<float>>(const double &c, std::complex<float> *output, int n) {
        VectorMulConstantInplace((float)c, (float *)output, 2 * n);
    }

    template <>
    void VectorMulConstantInplace<double, std::complex<double>>(const double &c, std::complex<double> *output, int n) {
        VectorMulConstantInplace(c, (double *)output, 2 * n);
    }

    template <>
    void VectorMulConstantInplace<std::complex<float>, std::complex<float>>(const std::complex<float> &c, std::complex<float> *output, int n) {
        ippsMulC_32fc_I(*((Ipp32fc *)&c), (Ipp32fc *)output, n);
    }

    template <>
    void VectorMulConstantInplace<std::complex<double>, std::complex<double>>(const std::complex<double> &c, std::complex<double> *output, int n) {
        ippsMulC_64fc_I(*((Ipp64fc *)&c), (Ipp64fc *)output, n);
    }

    template <>
    void VectorMulConstant<float>(const float *x, const float &c, float *output, int n) {
        ippsMulC_32f(x, c, output, n);
    }

    template <>
    void VectorMulConstant<double>(const double *x, const double &c, double *output, int n) {
        ippsMulC_64f(x, c, output, n);
    }

    template <>
    void VectorMulInplace<float, float>(const float *x, float *output, int n) {
        ippsMul_32f_I(x, output, n);
    }

    template <>
    void VectorMulInplace<double, double>(const double *x, double *output, int n) {
        ippsMul_64f_I(x, output, n);
    }

    template <>
    void VectorMulInplace<float, std::complex<float>>(const float *x, std::complex<float> *output, int n) {
        ippsMul_32f32fc_I(x, (Ipp32fc *)output, n);
    }

    template <>
    void VectorMulInplace<double, std::complex<double>>(const double *x, std::complex<double> *output, int n) {
        // ippsMul_64f64fc_Iが無いので自前
        for (int i = 0; i < n; i++) {
            output[i] *= x[i];
        }
    }

    template <>
    void VectorMulInplace<std::complex<float>, std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
        ippsMul_32fc_I((Ipp32fc *)x, (Ipp32fc *)output, n);
    }

    template <>
    void VectorMulInplace<std::complex<double>, std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
        ippsMul_64fc_I((Ipp64fc *)x, (Ipp64fc *)output, n);
    }

    template <>
    void VectorMul<float, float>(const float *x, const float *y, float *output, int n) {
        ippsMul_32f(x, y, output, n);
    }

    template <>
    void VectorMul<double, double>(const double *x, const double *y, double *output, int n) {
        ippsMul_64f(x, y, output, n);
    }

    template <>
    void VectorMul<float, std::complex<float>>(const float *x, const std::complex<float> *y, std::complex<float> *output, int n) {
        ippsMul_32f32fc(x, (Ipp32fc *)y, (Ipp32fc *)output, n);
    }

    template <>
    void VectorMul<std::complex<float>, std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
        ippsMul_32fc((Ipp32fc *)x, (Ipp32fc *)y, (Ipp32fc *)output, n);
    }

    template <>
    void VectorMul<double, std::complex<double>>(const double *x, const std::complex<double> *y, std::complex<double> *output, int n) {
        // ippsMul_64f64fcが無いので自前
        for (int i = 0; i < n; i++) {
            output[i] = x[i] * y[i];
        }
    }

    template <>
    void VectorMul<std::complex<double>, std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
        ippsMul_64fc((Ipp64fc *)x, (Ipp64fc *)y, (Ipp64fc *)output, n);
    }

    template <>
    void VectorMulPermInplace<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
        ippsMulPerm_32f_I((const float *)x, (float *)output, 2 * n);
    }

    template <>
    void VectorMulPermInplace<std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
        ippsMulPerm_64f_I((const double *)x, (double *)output, 2 * n);
    }

    template <>
    void VectorMulConj<std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
        ippsMulByConj_32fc_A24((Ipp32fc *)x, (Ipp32fc *)y, (Ipp32fc *)output, n);
    }

    template <>
    void VectorMulConj<std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
        ippsMulByConj_64fc_A53((Ipp64fc *)x, (Ipp64fc *)y, (Ipp64fc *)output, n);
    }

    template <>
    void VectorAddConstantInplace<float>(const float &c, float *output, int n) {
        ippsAddC_32f_I(c, output, n);
    }

    template <>
    void VectorAddConstantInplace<double>(const double &c, double *output, int n) {
        ippsAddC_64f_I(c, output, n);
    }

    template <>
    void VectorAddInplace<float>(const float *x, float *output, int n) {
        ippsAdd_32f_I(x, output, n);
    }

    template <>
    void VectorAddInplace<double>(const double *x, double *output, int n) {
        ippsAdd_64f_I(x, output, n);
    }

    template <>
    void VectorAddInplace<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
        ippsAdd_32fc_I((Ipp32fc *)x, (Ipp32fc *)output, n);
    }

    template <>
    void VectorAddInplace<std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
        ippsAdd_64fc_I((Ipp64fc *)x, (Ipp64fc *)output, n);
    }

    template <>
    void VectorAdd<float>(const float *x, const float *y, float *output, int n) {
        ippsAdd_32f(x, y, output, n);
    }

    template <>
    void VectorAdd<double>(const double *x, const double *y, double *output, int n) {
        ippsAdd_64f(x, y, output, n);
    }

    template <>
    void VectorAdd<std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
        ippsAdd_32fc((Ipp32fc *)x, (Ipp32fc *)y, (Ipp32fc *)output, n);
    }

    template <>
    void VectorAdd<std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
        ippsAdd_64fc((Ipp64fc *)x, (Ipp64fc *)y, (Ipp64fc *)output, n);
    }


    template <>
    void VectorSubConstantRev<float>(const float *x, const float &c, float *output, int n) {
        ippsSubCRev_32f(x, c, output, n);
    }

    template <>
    void VectorDivInplace<float>(const float *x, float *output, int n) {
        ippsDiv_32f_I(x, output, n);
    }

    template <>
    void VectorDivInplace<double>(const double *x, double *output, int n) {
        ippsDiv_64f_I(x, output, n);
    }

    template <>
    void VectorMadInplace<float>(const float *x, const float *y, float *output, int n) {
        ippsAddProduct_32f(x, y, output, n);
    }

    template <>
    void VectorMadInplace<double>(const double *x, const double *y, double *output, int n) {
        ippsAddProduct_64f(x, y, output, n);
    }

    template <>
    void VectorMadInplace<std::complex<float>>(const std::complex<float> *x, const std::complex<float> *y, std::complex<float> *output, int n) {
        ippsAddProduct_32fc((Ipp32fc *)x, (Ipp32fc *)y, (Ipp32fc *)output, n);
    }

    template <>
    void VectorMadInplace<std::complex<double>>(const std::complex<double> *x, const std::complex<double> *y, std::complex<double> *output, int n) {
        ippsAddProduct_64fc((Ipp64fc *)x, (Ipp64fc *)y, (Ipp64fc *)output, n);
    }

	template <>
	void VectorMadConstantInplace<float>(const float *x, const float &c, float *output, int n) {
		ippsAddProductC_32f(x, c, output, n);
	}

    template <>
    void VectorMadConstantInplace<double>(const double *x, const double &c, double *output, int n) {
        ippsAddProductC_64f(x, c, output, n);
    }

    template <>
    void VectorPowConstant<float>(const float *x, const float &c, float *output, int n) {
        ippsPowx_32f_A24(x, c, output, n);
    }

    template <>
    void VectorPowConstant<double>(const double *x, const double &c, double *output, int n) {
        ippsPowx_64f_A53(x, c, output, n);
    }

    template <>
    void VectorSqrtInplace<float>(float *output, int n) {
        ippsSqrt_32f_I(output, n);
    }

    template <>
    void VectorSqrtInplace<double>(double *output, int n) {
        ippsSqrt_64f_I(output, n);
    }

    template <>
    void VectorNorm<std::complex<float>, float>(const std::complex<float> *x, float *output, int n) {
        ippsPowerSpectr_32fc((const Ipp32fc *)x, output, n);
    }

    template <>
    void VectorNorm<std::complex<double>, double>(const std::complex<double> *x, double *output, int n) {
        ippsPowerSpectr_64fc((const Ipp64fc *)x, output, n);
    }

    template <>
    float VectorNormDiffL1<float>(const float *x, const float *y, int n) {
        float output;
        ippsNormDiff_L1_32f(x, y, n, &output);
        return output;
    }

    template <>
    double VectorNormDiffL1<double>(const double *x, const double *y, int n) {
        double output;
        ippsNormDiff_L1_64f(x, y, n, &output);
        return output;
    }

    template <>
    float VectorNormDiffL2<float>(const float *x, const float *y, int n) {
        float output;
        ippsNormDiff_L2_32f(x, y, n, &output);
        return output;
    }

    template <>
    double VectorNormDiffL2<double>(const double *x, const double *y, int n) {
        double output;
        ippsNormDiff_L2_64f(x, y, n, &output);
        return output;
    }

    template <>
    float VectorNormDiffInf<float>(const float *x, const float *y, int n) {
        float output;
        ippsNormDiff_Inf_32f(x, y, n, &output);
        return output;
    }

    template <>
    double VectorNormDiffInf<double>(const double *x, const double *y, int n) {
        double output;
        ippsNormDiff_Inf_64f(x, y, n, &output);
        return output;
    }

    template <>
    float VectorLInf<float>(const float *x, int n) {
        float result = 0;
        ippsNorm_Inf_32f(x, n, &result);
        return result;
    }

    template <>
    double VectorLInf<double>(const double *x, int n) {
        double result = 0;
        ippsNorm_Inf_64f(x, n, &result);
        return result;
    }

    template <>
    float VectorL2<float>(const float *x, int n) {
        float result = 0;
        ippsNorm_L2_32f(x, n, &result);
        return result;
    }

    template <>
    float VectorL2<std::complex<float>>(const std::complex<float> *x, int n) {
        return VectorL2((float *)x, 2 * n);
    }

    template <>
    double VectorL2<double>(const double *x, int n) {
        double result = 0;
        ippsNorm_L2_64f(x, n, &result);
        return result;
    }

    template <>
    double VectorL2<std::complex<double>>(const std::complex<double> *x, int n) {
        return VectorL2((double *)x, 2 * n);
    }

    template <>
    float VectorL2Sqr<float>(const std::complex<float> *x, int n) {
        float result = 0;
        ippsNorm_L2_32f((const float *)x, n, &result);
        return result;
    }

    template <>
    float VectorSum<float>(const float *x, int n) {
        float result = 0;
        ippsSum_32f((const float *)x, n, &result, ippAlgHintNone);
        return result;
    }

    template <>
    double VectorSum<double>(const double *x, int n) {
        double result = 0;
        ippsSum_64f((const double *)x, n, &result);
        return result;
    }

    template <>
    void VectorInvInplace<float>(float *output, int n) {
        ippsDivCRev_32f_I(1, output, n);
    }

    template <>
    void VectorInvInplace<double>(double *output, int n) {
        // ippsDivCRev_64f_Iがないので自前
        for (int i = 0; i < n; i++) {
            output[i] = 1.0 / output[i];
        }
    }

    template <>
    void VectorSet<float>(const float &c, float *output, int n) {
        ippsSet_32f(c, output, n);
    }

    template <>
    void VectorSet<double>(const double &c, double *output, int n) {
        ippsSet_64f(c, output, n);
    }

    template <>
    void VectorSet<int>(const int &c, int *output, int n) {
        ippsSet_32s(c, output, n);
    }

    template <>
    void VectorDecimate<float>(const float *x, int src_n, float *output, int factor) {
        int dest_n = 0;
        int phase = 0;
        ippsSampleDown_32f(x, src_n, output, &dest_n, factor, &phase);
    }

    template <>
    void VectorDecimate<double>(const double *x, int src_n, double *output, int factor) {
        int dest_n = 0;
        int phase = 0;
        ippsSampleDown_64f(x, src_n, output, &dest_n, factor, &phase);
    }

    template <>
    void VectorInterpolate<float>(const float *x, int src_n, float *output, int factor) {
        int dest_n = 0;
        int phase = 0;
        ippsSampleUp_32f(x, src_n, output, &dest_n, factor, &phase);
    }

    template <>
    void VectorInterpolate<double>(const double *x, int src_n, double *output, int factor) {
        int dest_n = 0;
        int phase = 0;
        ippsSampleUp_64f(x, src_n, output, &dest_n, factor, &phase);
    }

    template <>
    void VectorInterpolateHold<float>(const float *x, int src_n, float *output, int factor) {
        int k = 0;
        for (int i = 0; i < src_n; i++) {
            for (int j = 0; j < factor; j++) {
                output[k++] = x[i];
            }
        }
    }

    template <>
    void VectorInterpolateHold<double>(const double *x, int src_n, double *output, int factor) {
        int k = 0;
        for (int i = 0; i < src_n; i++) {
            for (int j = 0; j < factor; j++) {
                output[k++] = x[i];
            }
        }
    }

    template <>
    void VectorReverseInplace<std::complex<float>>(std::complex<float> *output, int n) {
        ippsFlip_32fc_I((Ipp32fc *)output, n);
    }

    template <>
    void VectorReverseInplace<std::complex<double>>(std::complex<double> *output, int n) {
        ippsFlip_64fc_I((Ipp64fc *)output, n);
    }

    template <>
    void VectorReverse<float>(const float *x, float *output, int n) {
        ippsFlip_32f(x, output, n);
    }

    template <>
    void VectorReverse<double>(const double *x, double *output, int n) {
        ippsFlip_64f(x, output, n);
    }

    template <>
    void VectorReverse<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
        ippsFlip_32fc((Ipp32fc *)x, (Ipp32fc *)output, n);
    }

    template <>
    void VectorReverse<std::complex<double>>(const std::complex<double> *x, std::complex<double> *output, int n) {
        ippsFlip_64fc((Ipp64fc *)x, (Ipp64fc *)output, n);
    }

    template <>
    void VectorConjInplace<std::complex<float>>(std::complex<float> *output, int n) {
        ippsConj_32fc_I((Ipp32fc *)output, n);
    }

    template <>
    void VectorConjInplace<std::complex<double>>(std::complex<double> *output, int n) {
        ippsConj_64fc_I((Ipp64fc *)output, n);
    }

    template <>
    void VectorMove<float>(const float *x, float *output, int n) {
        ippsMove_32f(x, output, n);
    }

    template <>
    void VectorMove<double>(const double *x, double *output, int n) {
        ippsMove_64f(x, output, n);
    }

    template <>
    void VectorMove<std::complex<float>>(const std::complex<float> *x, std::complex<float> *output, int n) {
        ippsMove_32fc((Ipp32fc *)x, (Ipp32fc *)output, n);
    }

    template <>
    void VectorZero<float>(float *output, int n) {
        ippsZero_32f(output, n);
    }

    template <>
    void VectorZero<std::complex<float>>(std::complex<float> *output, int n) {
        ippsZero_32fc((Ipp32fc *)output, n);
    }

    template <>
    float VectorDot<float>(const float *x, const float *y, int n) {
        float result = 0;
        ippsDotProd_32f(x, y, n, &result);
        return result;
    }

    template <>
    double VectorDot<double>(const double *x, const double *y, int n) {
        double result = 0;
        ippsDotProd_64f(x, y, n, &result);
        return result;
    }

	template <>
	void VectorReplaceNanInplace<float>(const float &c, float *x, int n) {
		ippsReplaceNAN_32f_I(x, n, c);
	}

    template <>
    void VectorEnsureNonnegativeInplace<float>(float *x, int n) {
        ippsThreshold_32f_I(x, n, 0, ippCmpLess);
    }

    template <>
    void VectorEnsureNonnegativeInplace<double>(double *x, int n) {
        ippsThreshold_64f_I(x, n, 0, ippCmpLess);
    }

	template <>
	void VectorBothThresholdInplace<float>(const float &c, float *x, int n) {
		ippsThreshold_32f_I(x, n, -c, ippCmpLess);
		ippsThreshold_32f_I(x, n, c, ippCmpGreater);
	}

    template <>
    void VectorConvert<float, float>(const float *x, float *output, int n) {
        if (x != output) {
            VectorMove(x, output, n);
        }
    }

    template <>
    void VectorConvert<double, double>(const double *x, double *output, int n) {
        if (x != output) {
            VectorMove(x, output, n);
        }
    }

    template <>
    void VectorConvert<float, Float16>(const float *x, Float16 *output, int n) {
        ippsConvert_32f16f(x, (Ipp16f *)output, n, ippRndNear);
    }

    template <>
    void VectorConvert<Float16, float>(const Float16 *x, float *output, int n) {
        ippsConvert_16f32f((Ipp16f *)x, output, n);
    }

    template <>
    void VectorConvert<std::complex<float>, ComplexFloat16>(const std::complex<float> *x, ComplexFloat16 *output, int n) {
        VectorConvert((const float *)x, (Float16 *)output, 2 * n);
    }

    template <>
    void VectorConvert<ComplexFloat16, std::complex<float>>(const ComplexFloat16 *x, std::complex<float> *output, int n) {
        VectorConvert((const Float16 *)x, (float *)output, 2 * n);
    }

    template <>
    void VectorRealToComplex<float>(const float *x, const float *y, std::complex<float> *output, int n) {
        ippsRealToCplx_32f(x, y, (Ipp32fc *)output, n);
    }

    template <>
    void VectorRealToComplex<double>(const double *x, const double *y, std::complex<double> *output, int n) {
        ippsRealToCplx_64f(x, y, (Ipp64fc *)output, n);
    }

    template <>
    void VectorComplexToReal<float>(const std::complex<float> *x, float *output_real, float *output_imag, int n) {
        ippsCplxToReal_32fc((Ipp32fc *)x, output_real, output_imag, n);
    }

    template <>
    void VectorComplexToReal<double>(const std::complex<double> *x, double *output_real, double *output_imag, int n) {
        ippsCplxToReal_64fc((Ipp64fc *)x, output_real, output_imag, n);
    }

    template <>
    void VectorConvolve<float>(const float *x, int nx, const float *y, int ny, float *output) {
        int buffer_size = 0;
        ippsConvolveGetBufferSize(nx, ny, ipp32f, ippAlgAuto, &buffer_size);
        AlignedPodVector<Ipp8u> buffer(buffer_size);

        ippsConvolve_32f((const Ipp32f *)x, nx, (const Ipp32f *)y, ny, (Ipp32f *)output, ippAlgAuto, buffer.data());
    }

    template <>
    void VectorConvolve<double>(const double *x, int nx, const double *y, int ny, double *output) {
        int buffer_size = 0;
        ippsConvolveGetBufferSize(nx, ny, ipp64f, ippAlgAuto, &buffer_size);
        AlignedPodVector<Ipp8u> buffer(buffer_size);

        ippsConvolve_64f((const Ipp64f *)x, nx, (const Ipp64f *)y, ny, (Ipp64f *)output, ippAlgAuto, buffer.data());
    }
}
