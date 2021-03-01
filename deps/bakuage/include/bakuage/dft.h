
#ifndef bakuage_fft_h
#define bakuage_fft_h

#include <complex>
#include <memory>
#include <unordered_map>

// FFTWと互換性のあるFFTを、thread safeで、バックエンドに依存させずに使えるようにするためのもの
// ライセンスとかを考えるとサーバーサイドの場合はFFTW or IPP、クライアントサイドはIPPなので、
// IPPバックエンドで実装する
// 本当にスピードが重要な場合は使わない

namespace bakuage {
    class FftMemoryBuffer {
    public:
        FftMemoryBuffer(int size = 0);
        virtual ~FftMemoryBuffer();
        
        FftMemoryBuffer(const FftMemoryBuffer& x);
        FftMemoryBuffer(FftMemoryBuffer&& x);
        FftMemoryBuffer& operator=(const FftMemoryBuffer& x);
        FftMemoryBuffer& operator=(FftMemoryBuffer&& x);
        
        int size() { return size_; }
        void *data() { return data_; }
		const void *data() const { return data_; }
    private:
        int size_;
        void *data_;
    };
    
    template <class Float>
    class Dft {};
    
    template <>
    class Dft<float> {
    public:
        typedef float Float;
        
        Dft(int len);
        virtual ~Dft() {}
        void Forward(const Float *input, Float *output);
        void Backward(const Float *input, Float *output);
    private:
        const void *dft_ptr_;
        FftMemoryBuffer work_;
    };
    
    template <>
    class Dft<double> {
    public:
        typedef double Float;
        
        Dft(int len);
        virtual ~Dft() {}
        void Forward(const Float *input, Float *output);
        void Backward(const Float *input, Float *output);
    private:
        const void *dft_ptr_;
        FftMemoryBuffer work_;
    };
    
    template <class Float>
    class Dft2D {};
    
    template <>
    class Dft2D<float> {
    public:
        typedef float Float;
        
        Dft2D(int size0, int size1);
        virtual ~Dft2D() {}
        void Forward(const Float *input, Float *output);
        void Backward(const Float *input, Float *output);
    private:
        const void *dft_ptr_;
        FftMemoryBuffer work_;
    };
    
    template <class Float>
    class Dct2D {};
    
    template <>
    class Dct2D<float> {
    public:
        typedef float Float;
        
        Dct2D(int size0, int size1);
        virtual ~Dct2D() {}
        void Forward(const Float *input, Float *output);
    private:
        const void *dct_ptr_;
        FftMemoryBuffer work_;
    };
    
    template <class Float>
    class RealDft {};
        
    template <>
    class RealDft<float> {
    public:
        typedef float Float;
        
        RealDft(int len, bool no_internal_work = false);
        virtual ~RealDft() {}
        void Forward(const Float *input, Float *output, void *work) const;
        void ForwardPerm(const Float *input, Float *output, void *work) const;
        void ForwardPack(const Float *input, Float *output, void *work) const;
        void Backward(const Float *input, Float *output, void *work) const;
        void BackwardPerm(const Float *input, Float *output, void *work) const;
        void BackwardPack(const Float *input, Float *output, void *work) const;
        void Forward(const Float *input, Float *output) {
            Forward(input, output, work_.data());
        }
        void ForwardPerm(const Float *input, Float *output) {
            ForwardPerm(input, output, work_.data());
        }
        void ForwardPack(const Float *input, Float *output) {
            ForwardPack(input, output, work_.data());
        }
        void Backward(const Float *input, Float *output) {
            Backward(input, output, work_.data());
        }
        void BackwardPerm(const Float *input, Float *output) {
            BackwardPerm(input, output, work_.data());
        }
        void BackwardPack(const Float *input, Float *output) {
            BackwardPack(input, output, work_.data());
        }
        size_t work_size() const;
    private:
        const void *dft_ptr_;
        const void *fft_ptr_;
        FftMemoryBuffer work_;
    };
    
    template <>
    class RealDft<double> {
    public:
        typedef double Float;
        
        RealDft(int len, bool no_internal_work = false);
        virtual ~RealDft() {}
        void Forward(const Float *input, Float *output, void *work) const;
        void ForwardPerm(const Float *input, Float *output, void *work) const;
        void ForwardPack(const Float *input, Float *output, void *work) const;
        void Backward(const Float *input, Float *output, void *work) const;
        void BackwardPerm(const Float *input, Float *output, void *work) const;
        void BackwardPack(const Float *input, Float *output, void *work) const;
        void Forward(const Float *input, Float *output) {
            Forward(input, output, work_.data());
        }
        void ForwardPerm(const Float *input, Float *output) {
            ForwardPerm(input, output, work_.data());
        }
        void ForwardPack(const Float *input, Float *output) {
            ForwardPack(input, output, work_.data());
        }
        void Backward(const Float *input, Float *output) {
            Backward(input, output, work_.data());
        }
        void BackwardPerm(const Float *input, Float *output) {
            BackwardPerm(input, output, work_.data());
        }
        void BackwardPack(const Float *input, Float *output) {
            BackwardPack(input, output, work_.data());
        }
        size_t work_size() const;
    private:
        const void *dft_ptr_;
        const void *fft_ptr_;
        FftMemoryBuffer work_;
    };
    
    // 異なるDft間で共有される
    class ThreadLocalDftWork {
    public:
        static ThreadLocalDftWork &GetThreadInstance() {
            static thread_local ThreadLocalDftWork instance;
            return instance;
        }
        void Reserve(int size) {
            if (!dft_work_data_ || dft_work_data_->size() < size) {
                dft_work_data_ = std::unique_ptr<bakuage::FftMemoryBuffer>(new bakuage::FftMemoryBuffer(size));
            }
        }
        void *work() {
            return dft_work_data_->data();
        }
    private:
        std::unique_ptr<bakuage::FftMemoryBuffer> dft_work_data_;
    };
    
    template <class Dft>
    class ThreadLocalDftPool {
    public:
        // thread_localの呼び出しは重いのでキャッシュしてなるべく減らす
        ThreadLocalDftPool(): work_(&ThreadLocalDftWork::GetThreadInstance()) {}
        static ThreadLocalDftPool &GetThreadInstance() {
            static thread_local ThreadLocalDftPool instance;
            return instance;
        }
        Dft *Get(int len) {
            auto found = dfts_.find(len);
            if (found == dfts_.end()) {
                dfts_.emplace(len, std::unique_ptr<Dft>(new Dft(len, true)));
                Dft *result = dfts_[len].get();
                work_->Reserve(result->work_size());
                return result;
            } else {
                return found->second.get();
            }
        }
        void *work() {
            return work_->work();
        }
    private:
        ThreadLocalDftWork *work_;
        std::unordered_map<int, std::unique_ptr<Dft>> dfts_;
    };

	static_assert(2 * sizeof(float) == sizeof(std::complex<float>), "2 * sizeof(float) == sizeof(std::complex<float>)");
	static_assert(2 * sizeof(double) == sizeof(std::complex<double>), "2 * sizeof(double) == sizeof(std::complex<double>)");
}

#endif /* fft_h */
