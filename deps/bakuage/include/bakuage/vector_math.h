#ifndef BAKUAGE_BAKUAGE_VECTOR_MATH_H
#define BAKUAGE_BAKUAGE_VECTOR_MATH_H

// vector同士の掛け算とかを高速に行う
// IPPに依存している。
// 実装はIPPのラッパーだが、テンプレートで使えるようにしている
// たくさんあるので、必要になったときに追加するスタイルで

// インターフェース設計
// input -> outputの順番で引数を指定 (like googleコーディング規約)
// inplaceは別で用意する
// 引数はポインタで指定
// Iteratorのbg, edではなく個数で指定する

#include <complex>

namespace bakuage {
#pragma pack(push, 1)
    // use struct instead of typedef to distinguish int and float in template
    struct Float16 { signed short as_int; };
    typedef Float16 ComplexFloat16[2];
#pragma pack(pop)

    template <class T>
    struct NormType {};
    template <>
    struct NormType<float> { typedef float Type; };
    template <>
    struct NormType<double> { typedef double Type; };
    template <>
    struct NormType<std::complex<float>> { typedef float Type; };
    template <>
    struct NormType<std::complex<double>> { typedef double Type; };

    template <class Float, class Float2>
    void VectorMulConstantInplace(const Float &c, Float2 *output, int n);

    template <class Float>
    void VectorMulConstant(const Float *x, const Float &c, Float *output, int n);

    template <class Float, class Float2>
    void VectorMulInplace(const Float *x, Float2 *output, int n);

    template <class Float, class Float2>
    void VectorMul(const Float *x, const Float2 *y, Float2 *output, int n);

    template <class Float>
    void VectorMulPermInplace(const Float *x, Float *output, int n);

    template <class Float>
    void VectorMulConj(const Float *x, const Float *y, Float *output, int n);

    template <class Float>
    void VectorAddConstantInplace(const Float &c, Float *output, int n);

    template <class Float>
    void VectorAddInplace(const Float *x, Float *output, int n);

    template <class Float>
    void VectorAdd(const Float *x, const Float *y, Float *output, int n);

    template <class Float>
    void VectorSubConstantRev(const Float *x, const Float &c, Float *output, int n);

    template <class Float>
    void VectorDivInplace(const Float *x, Float *output, int n);

    template <class Float>
    void VectorMadInplace(const Float *x, const Float *y, Float *output, int n);

	template <class Float>
	void VectorMadConstantInplace(const Float *x, const Float &c, Float *output, int n);

    template <class Float>
    void VectorPowConstant(const Float *x, const Float &c, Float *output, int n);

    template <class Float>
    void VectorSqrtInplace(Float *output, int n);

    // output = |x|^2
    template <class Float, class Float2>
    void VectorNorm(const Float *x, Float2 *output, int n);

    template <class Float>
    Float VectorNormDiffL1(const Float *x, const Float *y, int n);

    template <class Float>
    Float VectorNormDiffL2(const Float *x, const Float *y, int n);

    template <class Float>
    Float VectorNormDiffInf(const Float *x, const Float *y, int n);

    template <class Float>
    Float VectorLInf(const Float *x, int n);

    template <class Float>
    typename NormType<Float>::Type VectorL2(const Float *x, int n);

    template <class Float>
    Float VectorL2Sqr(const std::complex<Float> *x, int n);

    template <class Float>
    Float VectorSum(const Float *x, int n);

    template <class Float>
    void VectorInvInplace(Float *output, int n);

    template <class Float>
    void VectorSet(const Float &c, Float *output, int n);

    template <class Float>
    void VectorDecimate(const Float *x, int src_n, Float *output, int factor);

    template <class Float>
    void VectorInterpolate(const Float *x, int src_n, Float *output, int factor);

    template <class Float>
    void VectorInterpolateHold(const Float *x, int src_n, Float *output, int factor);

    template <class Float>
    void VectorReverseInplace(Float *output, int n);

    template <class Float>
    void VectorReverse(const Float *x, Float *output, int n);

    template <class Float>
    void VectorConjInplace(Float *output, int n);

    // memmoveより速いわけではないので、使う必要なし
    template <class Float>
    void VectorMove(const Float *x, Float *output, int n);

    template <class Float>
    void VectorZero(Float *output, int n);

    template <class Float>
    Float VectorDot(const Float *x, const Float *y, int n);

	template <class Float>
	void VectorReplaceNanInplace(const Float &c, Float *x, int n);

    template <class Float>
    void VectorEnsureNonnegativeInplace(Float *x, int n);

	template <class Float>
	void VectorBothThresholdInplace(const Float &c, Float *x, int n);

	template <class Float>
	void VectorSanitizeInplace(const Float &c, Float *x, int n) {
		VectorBothThresholdInplace<Float>(c, x, n);
		VectorReplaceNanInplace<Float>(0, x, n);
	}

    template <class Float, class Float2>
    void VectorConvert(const Float *x, Float2 *output, int n);

    template <class Float>
    void VectorRealToComplex(const Float *x, const Float *y, std::complex<Float> *output, int n);

    template <class Float>
    void VectorComplexToReal(const std::complex<Float> *x, Float *output_real, Float *output_imag, int n);

    template <class Float>
    void VectorConvolve(const Float *x, int nx, const Float *y, int ny, Float *output);
}

#endif /* vector_math_h */
