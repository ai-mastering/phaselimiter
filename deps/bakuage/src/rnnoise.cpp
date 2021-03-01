#include "bakuage/rnnoise.h"

#include <boost/dll/import.hpp>
#include <boost/function.hpp>

namespace {
    struct RnnoiseLibraryImpl {
        RnnoiseLibraryImpl(const char *librnnoise_path): library_(librnnoise_path) {
            if (!library_) {
                throw std::logic_error(std::string("failed to load ") + librnnoise_path);
            }
            rnnoise_create_ = library_.get<void *()>("rnnoise_create");
            rnnoise_destroy_ = library_.get<void(void *)>("rnnoise_destroy");
            rnnoise_process_frame_ = library_.get<float(void *, float *, const float *, int, float)>("rnnoise_process_frame");
            rnnoise_process_frame_multi_available_ = library_.has("rnnoise_process_frame_multi");
            if (rnnoise_process_frame_multi_available_) {
                rnnoise_process_frame_multi_ = library_.get<float(int, void **, float **, const float **, int, float)>("rnnoise_process_frame_multi");
            }
        }
        
        boost::dll::shared_library library_;
        boost::function<void *()> rnnoise_create_;
        boost::function<void(void *)> rnnoise_destroy_;
        boost::function<float(void *, float *, const float *, int, float)> rnnoise_process_frame_;
        bool rnnoise_process_frame_multi_available_;
        boost::function<float(int, void **, float **, const float **, int, float)> rnnoise_process_frame_multi_;
    };
    
    struct RnnoiseLibraryImplDeleter {
        void operator()(void *ptr){
            delete (RnnoiseLibraryImpl *)ptr;
        }
    };
}

namespace bakuage {
    RnnoiseLibrary::RnnoiseLibrary(const char *librnnoise_path): impl_(std::shared_ptr<void>(new RnnoiseLibraryImpl(librnnoise_path), RnnoiseLibraryImplDeleter())) {
    }
    void *RnnoiseLibrary::rnnoise_create() const {
        return ((RnnoiseLibraryImpl *)impl_.get())->rnnoise_create_();
    }
    void RnnoiseLibrary::rnnoise_destroy(void *st) const {
        ((RnnoiseLibraryImpl *)impl_.get())->rnnoise_destroy_(st);
    }
    float RnnoiseLibrary::rnnoise_process_frame(void *st, float *out, const float *in, int pitch_filter_enabled) const {
        return ((RnnoiseLibraryImpl *)impl_.get())->rnnoise_process_frame_(st, out, in, pitch_filter_enabled, 1.0);
    }
    float RnnoiseLibrary::rnnoise_process_frame_multi(int num_channels, void **st, float **out, const float **in, int pitch_filter_enabled) const {
        const auto impl = (RnnoiseLibraryImpl *)impl_.get();
        if (impl->rnnoise_process_frame_multi_available_) {
            return impl->rnnoise_process_frame_multi_(num_channels, st, out, in, pitch_filter_enabled, 1.0);
        } else {
            double vad_prob_sum = 0;
            for (int ch = 0; ch < num_channels; ch++) {
                vad_prob_sum += impl->rnnoise_process_frame_(st[ch], out[ch], in[ch], pitch_filter_enabled, 1.0);
            }
            return vad_prob_sum / num_channels;
        }
    }
}
