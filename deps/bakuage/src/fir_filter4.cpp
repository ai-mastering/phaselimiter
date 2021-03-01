#include "bakuage/fir_filter4.h"
#include "ipp.h"

namespace {
    template <class Float>
    struct Dispatch {};
    
    template <>
    struct Dispatch<float> {
        typedef IppsFIRSpec_32f SpecType;
        typedef Ipp32f DataType;
        static constexpr IppDataType kDataType = ipp32f;
        static IppStatus MRInit(const DataType *pTaps, int tapsLen, int upFactor, int upPhase, int downFactor, int downPhase, SpecType *pSpec) {
            return ippsFIRMRInit_32f(pTaps, tapsLen, upFactor, upPhase, downFactor, downPhase, pSpec);
        }
        static IppStatus MR(const DataType *pSrc, DataType *pDst, int numIters, SpecType *pSpec, const DataType *pDlySrc, DataType *pDlyDst, Ipp8u *pBuf) {
            return ippsFIRMR_32f(pSrc, pDst, numIters, pSpec, pDlySrc, pDlyDst, pBuf);
        }
        static DataType *Malloc(int size) {
            return ippsMalloc_32f(size);
        }
    };
    
    template <>
    struct Dispatch<double> {
        typedef IppsFIRSpec_64f SpecType;
        typedef Ipp64f DataType;
        static constexpr IppDataType kDataType = ipp64f;
        static IppStatus MRInit(const DataType *pTaps, int tapsLen, int upFactor, int upPhase, int downFactor, int downPhase, SpecType *pSpec) {
            return ippsFIRMRInit_64f(pTaps, tapsLen, upFactor, upPhase, downFactor, downPhase, pSpec);
        }
        static IppStatus MR(const DataType *pSrc, DataType *pDst, int numIters, SpecType *pSpec, const DataType *pDlySrc, DataType *pDlyDst, Ipp8u *pBuf) {
            return ippsFIRMR_64f(pSrc, pDst, numIters, pSpec, pDlySrc, pDlyDst, pBuf);
        }
        static DataType *Malloc(int size) {
            return ippsMalloc_64f(size);
        }
    };
    
    template <>
    struct Dispatch<std::complex<float>> {
        typedef IppsFIRSpec_32fc SpecType;
        typedef Ipp32fc DataType;
        // static constexpr IppDataType kDataType = ipp32fc;
        static IppStatus MRInit(const DataType *pTaps, int tapsLen, int upFactor, int upPhase, int downFactor, int downPhase, SpecType *pSpec) {
            return ippsFIRMRInit_32fc(pTaps, tapsLen, upFactor, upPhase, downFactor, downPhase, pSpec);
        }
        static IppStatus MR(const DataType *pSrc, DataType *pDst, int numIters, SpecType *pSpec, const DataType *pDlySrc, DataType *pDlyDst, Ipp8u *pBuf) {
            return ippsFIRMR_32fc(pSrc, pDst, numIters, pSpec, pDlySrc, pDlyDst, pBuf);
        }
        static DataType *Malloc(int size) {
            return ippsMalloc_32fc(size);
        }
    };
    
    template <>
    struct Dispatch<std::complex<double>> {
        typedef IppsFIRSpec_64fc SpecType;
        typedef Ipp64fc DataType;
        // static constexpr IppDataType kDataType = ipp64fc;
        static IppStatus MRInit(const DataType *pTaps, int tapsLen, int upFactor, int upPhase, int downFactor, int downPhase, SpecType *pSpec) {
            return ippsFIRMRInit_64fc(pTaps, tapsLen, upFactor, upPhase, downFactor, downPhase, pSpec);
        }
        static IppStatus MR(const DataType *pSrc, DataType *pDst, int numIters, SpecType *pSpec, const DataType *pDlySrc, DataType *pDlyDst, Ipp8u *pBuf) {
            return ippsFIRMR_64fc(pSrc, pDst, numIters, pSpec, pDlySrc, pDlyDst, pBuf);
        }
        static DataType *Malloc(int size) {
            return ippsMalloc_64fc(size);
        }
    };
}

namespace bakuage {
    template <class Float>
    struct FirFilter4Impl: public FirFilter4ImplBase {
        FirFilter4Impl(const Float *bg, const Float *ed, int up_factor, int down_factor): spec(nullptr), buffer(nullptr), delay(nullptr), delay2(nullptr), delay_idx(0), up_factor_(up_factor), down_factor_(down_factor) {
            const int fir_size = ed - bg;
            int spec_size = 0;
            int buf_size = 0;
            ippsFIRMRGetSize(fir_size, up_factor, down_factor, Dispatch<Float>::kDataType, &spec_size, &buf_size);
            
            spec = (typename Dispatch<Float>::SpecType *)ippsMalloc_8u(spec_size);
            std::memset(spec, 0, spec_size);
            buffer = ippsMalloc_8u(buf_size);
            std::memset(buffer, 0, buf_size);
            delay = Dispatch<Float>::Malloc(fir_size);
            std::memset(delay, 0, sizeof(typename Dispatch<Float>::DataType) * fir_size);
            delay2 = Dispatch<Float>::Malloc(fir_size);
            std::memset(delay2, 0, sizeof(typename Dispatch<Float>::DataType) * fir_size);
            
            Dispatch<Float>::MRInit(bg, fir_size, up_factor, 0, down_factor, 0, spec);
        }
        virtual ~FirFilter4Impl() {
            if (spec) ippsFree(spec);
            if (buffer) ippsFree(buffer);
            if (delay) ippsFree(delay);
        }
        
        void Clock(const Float *bg, const Float *ed, Float *output) {
            if (delay_idx % 2 == 0) {
                Dispatch<Float>::MR(bg, output, (ed - bg) / down_factor_, spec, delay, delay2, buffer);
            } else {
                Dispatch<Float>::MR(bg, output, (ed - bg) / down_factor_, spec, delay2, delay, buffer);
            }
            delay_idx++;
        }
        
    private:
        typename Dispatch<Float>::SpecType *spec;
        Ipp8u *buffer;
        typename Dispatch<Float>::DataType *delay;
        typename Dispatch<Float>::DataType *delay2;
        int delay_idx;
        int up_factor_;
        int down_factor_;
    };
    
    template<>
    void FirFilter4<float>::Clock(const float *bg, const float *ed, float *output) {
        ((FirFilter4Impl<float> *)impl_.get())->Clock(bg, ed, output);
    }
    
    template<>
    void FirFilter4<float>::PrepareFir(const float *bg, const float *ed, int up_factor, int down_factor) {
        impl_ = std::unique_ptr<FirFilter4ImplBase>(new FirFilter4Impl<float>(bg, ed, up_factor, down_factor));
    }
    
    template<>
    void FirFilter4<double>::Clock(const double *bg, const double *ed, double *output) {
        ((FirFilter4Impl<double> *)impl_.get())->Clock(bg, ed, output);
    }
    
    template<>
    void FirFilter4<double>::PrepareFir(const double *bg, const double *ed, int up_factor, int down_factor) {
        impl_ = std::unique_ptr<FirFilter4ImplBase>(new FirFilter4Impl<double>(bg, ed, up_factor, down_factor));
    }
   
}


