#include "bakuage/dft.h"

#include <cstring>
#include <iostream>
#include <mutex>
#include <unordered_map>
#include "ipp.h"
#include "bakuage/utils.h"

namespace {
    constexpr int ipp_verbose = 0;
    
    struct MyIppDftBase {
        MyIppDftBase(): workBufferSize(0) {}
        int workBufferSize;
    };
    
    struct MyIppR2CDft32: public MyIppDftBase {
        typedef float Float;
        typedef int Size;
        typedef std::hash<int> SizeHash;
        
        MyIppR2CDft32(int len): MyIppDftBase() {
            if (ipp_verbose) std::cerr << "ipp dft create " << len << std::endl;
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippsDFTGetSize_R_32f(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
			spec_buffer_ = bakuage::FftMemoryBuffer(specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippsDFTInit_R_32f(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, specPtr(), specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_RToCCS_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_RToPerm_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_RToPack_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_CCSToR_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_PermToR_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_PackToR_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }

		IppsDFTSpec_R_32f *specPtr() { return (IppsDFTSpec_R_32f *)spec_buffer_.data(); };
		const IppsDFTSpec_R_32f *specPtr() const { return (IppsDFTSpec_R_32f *)spec_buffer_.data(); };

		bakuage::FftMemoryBuffer spec_buffer_;
    };
    
    struct MyIppR2CFft32: public MyIppDftBase {
        typedef float Float;
        typedef int Size;
        typedef std::hash<int> SizeHash;
        
        MyIppR2CFft32(int len): MyIppDftBase() {
            if (ipp_verbose) std::cerr << "ipp dft create " << len << std::endl;
            const int order = bakuage::IntLog2(len);
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippsFFTGetSize_R_32f(order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
            spec_buffer_ = bakuage::FftMemoryBuffer(specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippsFFTInit_R_32f(&spec_ptr_, order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, (Ipp8u *)spec_buffer_.data(), specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToCCS_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPerm_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePermInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPerm_32f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPack_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePackInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPack_32f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_CCSToR_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PermToR_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPermInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PermToR_32f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PackToR_32f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPackInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PackToR_32f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }
        
        IppsFFTSpec_R_32f *specPtr() { return spec_ptr_; };
        const IppsFFTSpec_R_32f *specPtr() const { return spec_ptr_; };
        
        IppsFFTSpec_R_32f *spec_ptr_;
        bakuage::FftMemoryBuffer spec_buffer_;
    };
    
    struct MyIppR2CDft64: public MyIppDftBase {
        typedef double Float;
        typedef int Size;
        typedef std::hash<int> SizeHash;
        
        MyIppR2CDft64(int len): MyIppDftBase() {
            if (ipp_verbose) std::cerr << "ipp dft create " << len << std::endl;
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippsDFTGetSize_R_64f(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
            spec_buffer_ = bakuage::FftMemoryBuffer(specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippsDFTInit_R_64f(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, specPtr(), specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_RToCCS_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_RToPerm_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_RToPack_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_CCSToR_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_PermToR_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_PackToR_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }
        
        IppsDFTSpec_R_64f *specPtr() { return (IppsDFTSpec_R_64f *)spec_buffer_.data(); };
        const IppsDFTSpec_R_64f *specPtr() const { return (IppsDFTSpec_R_64f *)spec_buffer_.data(); };
        
        bakuage::FftMemoryBuffer spec_buffer_;
    };
    
    struct MyIppR2CFft64: public MyIppDftBase {
        typedef double Float;
        typedef int Size;
        typedef std::hash<int> SizeHash;
        
        MyIppR2CFft64(int len): MyIppDftBase() {
            if (ipp_verbose) std::cerr << "ipp dft create " << len << std::endl;
            const int order = bakuage::IntLog2(len);
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippsFFTGetSize_R_64f(order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
            spec_buffer_ = bakuage::FftMemoryBuffer(specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippsFFTInit_R_64f(&spec_ptr_, order, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, (Ipp8u *)spec_buffer_.data(), specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToCCS_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPerm_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePermInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPerm_64f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPack_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecutePackInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTFwd_RToPack_64f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_CCSToR_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPerm(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PermToR_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPermInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PermToR_64f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPack(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PackToR_64f(src, dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInvPackInplace(Float *src_dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsFFTInv_PackToR_64f_I(src_dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }
        
        IppsFFTSpec_R_64f *specPtr() { return spec_ptr_; };
        const IppsFFTSpec_R_64f *specPtr() const { return spec_ptr_; };
        
        IppsFFTSpec_R_64f *spec_ptr_;
        bakuage::FftMemoryBuffer spec_buffer_;
    };
    
    struct MyIppC2CDft32: public MyIppDftBase {
        typedef float Float;
        typedef int Size;
        typedef std::hash<int> SizeHash;
        
        MyIppC2CDft32(int len): MyIppDftBase() {
            if (ipp_verbose) std::cerr << "ipp dft create " << len << std::endl;
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippsDFTGetSize_C_32fc(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;

			spec_buffer_ = bakuage::FftMemoryBuffer(specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippsDFTInit_C_32fc(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, specPtr(), specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_CToC_32fc((const Ipp32fc *)src, (Ipp32fc *)dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_CToC_32fc((const Ipp32fc *)src, (Ipp32fc *)dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }

		IppsDFTSpec_C_32fc *specPtr() { return (IppsDFTSpec_C_32fc *)spec_buffer_.data(); };
		const IppsDFTSpec_C_32fc *specPtr() const { return (IppsDFTSpec_C_32fc *)spec_buffer_.data(); };

		bakuage::FftMemoryBuffer spec_buffer_;
    };
    
    struct MyIppC2CDft64: public MyIppDftBase {
        typedef double Float;
        typedef int Size;
        typedef std::hash<int> SizeHash;
        
        MyIppC2CDft64(int len): MyIppDftBase() {
            if (ipp_verbose) std::cerr << "ipp dft create " << len << std::endl;
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippsDFTGetSize_C_64fc(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
            spec_buffer_ = bakuage::FftMemoryBuffer(specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippsDFTInit_C_64fc(len, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, specPtr(), specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTFwd_CToC_64fc((const Ipp64fc *)src, (Ipp64fc *)dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippsDFTInv_CToC_64fc((const Ipp64fc *)src, (Ipp64fc *)dest, specPtr(), workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }
        
        IppsDFTSpec_C_64fc *specPtr() { return (IppsDFTSpec_C_64fc *)spec_buffer_.data(); };
        const IppsDFTSpec_C_64fc *specPtr() const { return (IppsDFTSpec_C_64fc *)spec_buffer_.data(); };
        
        bakuage::FftMemoryBuffer spec_buffer_;
    };
    
    struct Size2D {
        std::size_t operator()(const Size2D& key) const {
            std::hash<int> h;
            return h(size0) + h(size1);
        }
        bool operator == (const Size2D &other) const {
            return size0 == other.size0 && size1 == other.size1;
        }
        int size0;
        int size1;
    };
    
    struct MyIppC2CDft2D32: public MyIppDftBase {
        typedef float Float;
        typedef Size2D Size;
        typedef Size2D SizeHash;
        
        MyIppC2CDft2D32(const Size &len): MyIppDftBase(), specPtr(nullptr), size_(len) {
            if (ipp_verbose) std::cerr << "ipp dft create " << size_.size0 << " " << size_.size1 << std::endl;
            IppiSize size;
            size.width = len.size0;
            size.height = len.size1;
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippiDFTGetSize_C_32fc(size, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
            specPtr = (IppiDFTSpec_C_32fc *)ippsMalloc_8u(specSize);
            if (specPtr) std::memset(specPtr, 0, specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippiDFTInit_C_32fc(size, IPP_FFT_NODIV_BY_ANY, ippAlgHintNone, specPtr, specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippiDFTFwd_CToC_32fc_C1R((const Ipp32fc *)src, stride_bytes(), (Ipp32fc *)dest, stride_bytes(), specPtr, workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
        void ExecuteInv(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippiDFTInv_CToC_32fc_C1R((const Ipp32fc *)src, stride_bytes(), (Ipp32fc *)dest, stride_bytes(), specPtr, workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        int stride_bytes() const {
            return 2 * sizeof(Float) * size_.size0;
        }
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }
        
        IppiDFTSpec_C_32fc *specPtr;
        Size size_;
    };
    
    struct MyIppDct2D32: public MyIppDftBase {
        typedef float Float;
        typedef Size2D Size;
        typedef Size2D SizeHash;
        
        MyIppDct2D32(const Size &len): MyIppDftBase(), specPtr(nullptr), size_(len) {
            if (ipp_verbose) std::cerr << "ipp dft create " << size_.size0 << " " << size_.size1 << std::endl;
            IppiSize size;
            size.width = len.size0;
            size.height = len.size1;
            
            int specSize = 0;
            int specBufferSize = 0;
            CheckResult(ippiDCTFwdGetSize_32f(size, &specSize, &specBufferSize, &workBufferSize));
            
            if (ipp_verbose) std::cerr << "ipp dft get size " << specSize << " " << specBufferSize << " " << workBufferSize << std::endl;
            
            specPtr = (IppiDCTFwdSpec_32f *)ippsMalloc_8u(specSize);
            if (specPtr) std::memset(specPtr, 0, specSize);
            
            Ipp8u *specBuffer = specBufferSize ? ippsMalloc_8u(specBufferSize) : nullptr;
            if (specBuffer) std::memset(specBuffer, 0, specBufferSize);
            
            if (ipp_verbose) std::cerr << "ipp dft allocated" << std::endl;
            
            CheckResult(ippiDCTFwdInit_32f(specPtr, size, specBuffer));
            
            if (specBuffer) ippsFree(specBuffer);
            
            if (ipp_verbose) std::cerr << "ipp dft initialized" << std::endl;
        }
        void Execute(const Float *src, Float *dest, void *work) const {
            if (ipp_verbose) std::cerr << "ipp dft execute" << std::endl;
            CheckResult(ippiDCTFwd_32f_C1R(src, stride_bytes(), dest, stride_bytes(), specPtr, workBufferSize ? (Ipp8u *)work : nullptr));
            if (ipp_verbose) std::cerr << "ipp dft finished" << std::endl;
        }
    private:
        int stride_bytes() const {
            return 1 * sizeof(Float) * size_.size0;
        }
        void CheckResult(IppStatus result) const {
            if (result != ippStsNoErr) {
                std::cerr << "ipp dft error " << result << std::endl;
            }
        }
        
        IppiDCTFwdSpec_32f *specPtr;
        Size size_;
    };
    
    template <class T>
    struct MyIppDftLibrary {
        MyIppDftLibrary() {}
        static MyIppDftLibrary &GetInstance() {
            static MyIppDftLibrary instance;
            return instance;
        }
        const T *get(const typename T::Size &len) {
            std::lock_guard<std::mutex> lock(mtx_);
            
            const auto found = dfts_.find(len);
            if (found != dfts_.end()) {
                return found->second;
            }
            const auto dft = new T(len);
            dfts_.insert(std::pair<typename T::Size, T *>(len, dft));
            return dft;
        }
    private:
        std::mutex mtx_;
        std::unordered_map<typename T::Size, T *, typename T::SizeHash> dfts_;
    };
    
}

namespace bakuage {
    FftMemoryBuffer::FftMemoryBuffer(int size): size_(size), data_(size ? ippsMalloc_8u(size) : nullptr) {
		if (size_) std::memset(data_, 0, size_);
	}

    FftMemoryBuffer::~FftMemoryBuffer() {
        if (data_) ippsFree(data_);
    }
    
    FftMemoryBuffer::FftMemoryBuffer(const FftMemoryBuffer& x): size_(x.size_), data_(size_ ? ippsMalloc_8u(size_) : nullptr) {
		if (size_) {
			std::memcpy(data_, x.data_, size_);
		}
    }
    FftMemoryBuffer::FftMemoryBuffer(FftMemoryBuffer&& x): size_(x.size_), data_(x.data_) {
        x.size_ = 0;
        x.data_ = nullptr;
    }
    FftMemoryBuffer& FftMemoryBuffer::operator=(const FftMemoryBuffer& x) {
        size_ = x.size_;
        data_ = size_ ? ippsMalloc_8u(size_) : nullptr;
		if (size_) {
			std::memcpy(data_, x.data_, size_);
		}
        return *this;
    }
    FftMemoryBuffer& FftMemoryBuffer::operator=(FftMemoryBuffer&& x) {
        size_ = x.size_;
        data_ = x.data_;
        x.size_ = 0;
        x.data_ = nullptr;
        return *this;
    }
    
    Dft<float>::Dft(int len) {
        const auto dft = MyIppDftLibrary<MyIppC2CDft32>::GetInstance().get(len);
        dft_ptr_ = (void *)dft;
        work_ = FftMemoryBuffer(dft->workBufferSize);
    }
    void Dft<float>::Forward(const Float *input, Float *output) {
        const auto dft = (MyIppC2CDft32 *)dft_ptr_;
        dft->Execute(input, output, work_.data());
    }
    void Dft<float>::Backward(const Float *input, Float *output) {
        const auto dft = (MyIppC2CDft32 *)dft_ptr_;
        dft->ExecuteInv(input, output, work_.data());
    }
    
    Dft<double>::Dft(int len) {
        const auto dft = MyIppDftLibrary<MyIppC2CDft64>::GetInstance().get(len);
        dft_ptr_ = (void *)dft;
        work_ = FftMemoryBuffer(dft->workBufferSize);
    }
    void Dft<double>::Forward(const Float *input, Float *output) {
        const auto dft = (MyIppC2CDft64 *)dft_ptr_;
        dft->Execute(input, output, work_.data());
    }
    void Dft<double>::Backward(const Float *input, Float *output) {
        const auto dft = (MyIppC2CDft64 *)dft_ptr_;
        dft->ExecuteInv(input, output, work_.data());
    }
    
    Dft2D<float>::Dft2D(int size0, int size1) {
        Size2D size;
        size.size0 = size0;
        size.size1 = size1;
        const auto dft = MyIppDftLibrary<MyIppC2CDft2D32>::GetInstance().get(size);
        dft_ptr_ = (void *)dft;
        work_ = FftMemoryBuffer(dft->workBufferSize);
    }
    void Dft2D<float>::Forward(const Float *input, Float *output) {
        const auto dft = (MyIppC2CDft2D32 *)dft_ptr_;
        dft->Execute(input, output, work_.data());
    }
    void Dft2D<float>::Backward(const Float *input, Float *output) {
        const auto dft = (MyIppC2CDft2D32 *)dft_ptr_;
        dft->ExecuteInv(input, output, work_.data());
    }
    
    Dct2D<float>::Dct2D(int size0, int size1) {
        Size2D size;
        size.size0 = size0;
        size.size1 = size1;
        const auto dct = MyIppDftLibrary<MyIppDct2D32>::GetInstance().get(size);
        dct_ptr_ = (void *)dct;
        work_ = FftMemoryBuffer(dct->workBufferSize);
    }
    void Dct2D<float>::Forward(const Float *input, Float *output) {
        const auto dct = (MyIppDct2D32 *)dct_ptr_;
        dct->Execute(input, output, work_.data());
    }
    
    RealDft<float>::RealDft(int len, bool no_internal_work) {
        const auto dft = MyIppDftLibrary<MyIppR2CDft32>::GetInstance().get(len);
        const auto fft = MyIppDftLibrary<MyIppR2CFft32>::GetInstance().get(len);
        dft_ptr_ = (void *)dft;
        fft_ptr_ = (void *)fft;
        if (!no_internal_work) {
            work_ = FftMemoryBuffer(work_size());
        }
    }
    void RealDft<float>::Forward(const Float *input, Float *output, void *work_data) const {
        const auto dft = (MyIppR2CDft32 *)dft_ptr_;
        dft->Execute(input, output, work_data);
    }
    void RealDft<float>::ForwardPerm(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft32 *)fft_ptr_;
            fft->ExecutePermInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft32 *)dft_ptr_;
            dft->ExecutePerm(input, output, work_data);
        }
    }
    void RealDft<float>::ForwardPack(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft32 *)fft_ptr_;
            fft->ExecutePackInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft32 *)dft_ptr_;
            dft->ExecutePack(input, output, work_data);
        }
    }
    void RealDft<float>::Backward(const Float *input, Float *output, void *work_data) const {
        const auto dft = (MyIppR2CDft32 *)dft_ptr_;
        dft->ExecuteInv(input, output, work_data);
    }
    void RealDft<float>::BackwardPerm(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft32 *)fft_ptr_;
            fft->ExecuteInvPermInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft32 *)dft_ptr_;
            dft->ExecuteInvPerm(input, output, work_data);
        }
    }
    void RealDft<float>::BackwardPack(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft32 *)fft_ptr_;
            fft->ExecuteInvPackInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft32 *)dft_ptr_;
            dft->ExecuteInvPack(input, output, work_data);
        }
    }
    size_t RealDft<float>::work_size() const {
        const auto dft = (MyIppR2CDft32 *)dft_ptr_;
        const auto fft = (MyIppR2CFft32 *)fft_ptr_;
        return (std::max)(dft->workBufferSize, fft->workBufferSize);
    }
    
    RealDft<double>::RealDft(int len, bool no_internal_work) {
        const auto dft = MyIppDftLibrary<MyIppR2CDft64>::GetInstance().get(len);
        const auto fft = MyIppDftLibrary<MyIppR2CFft64>::GetInstance().get(len);
        dft_ptr_ = (void *)dft;
        fft_ptr_ = (void *)fft;
        if (!no_internal_work) {
            work_ = FftMemoryBuffer(work_size());
        }
    }
    void RealDft<double>::Forward(const Float *input, Float *output, void *work_data) const {
        const auto dft = (MyIppR2CDft64 *)dft_ptr_;
        dft->Execute(input, output, work_data);
    }
    void RealDft<double>::ForwardPerm(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft64 *)fft_ptr_;
            fft->ExecutePermInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft64 *)dft_ptr_;
            dft->ExecutePerm(input, output, work_data);
        }
    }
    void RealDft<double>::ForwardPack(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft64 *)fft_ptr_;
            fft->ExecutePackInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft64 *)dft_ptr_;
            dft->ExecutePack(input, output, work_data);
        }
    }
    void RealDft<double>::Backward(const Float *input, Float *output, void *work_data) const {
        const auto dft = (MyIppR2CDft64 *)dft_ptr_;
        dft->ExecuteInv(input, output, work_data);
    }
    void RealDft<double>::BackwardPerm(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft64 *)fft_ptr_;
            fft->ExecuteInvPermInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft64 *)dft_ptr_;
            dft->ExecuteInvPerm(input, output, work_data);
        }
    }
    void RealDft<double>::BackwardPack(const Float *input, Float *output, void *work_data) const {
        if (input == output) {
            const auto fft = (MyIppR2CFft64 *)fft_ptr_;
            fft->ExecuteInvPackInplace(output, work_data);
        } else {
            const auto dft = (MyIppR2CDft64 *)dft_ptr_;
            dft->ExecuteInvPack(input, output, work_data);
        }
    }
    size_t RealDft<double>::work_size() const {
        const auto dft = (MyIppR2CDft64 *)dft_ptr_;
        const auto fft = (MyIppR2CFft64 *)fft_ptr_;
        return (std::max)(dft->workBufferSize, fft->workBufferSize);
    }
    
}
