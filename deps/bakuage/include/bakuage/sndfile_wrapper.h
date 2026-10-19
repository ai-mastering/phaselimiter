#ifndef BAKUAGE_BAKUAGE_SNDFILE_WRAPPER_H_
#define BAKUAGE_BAKUAGE_SNDFILE_WRAPPER_H_

#include "sndfile.h"
#include <stdexcept>
#include <string>

namespace bakuage {
	class SndfileWrapper {
	public:
	    SndfileWrapper(): f_(NULL) {}
        SndfileWrapper(const SndfileWrapper &) = delete;
        SndfileWrapper &operator=(const SndfileWrapper &) = delete;
        virtual ~SndfileWrapper() {
            close();
        }
        // Closing consumes the handle even if finalization fails. Destruction
        // is best effort; writers must check this result explicitly.
        int close() noexcept {
            SNDFILE *file = f_;
            f_ = NULL;
            return file ? sf_close(file) : 0;
        }
        void CloseChecked() {
            const int error = close();
            if (error) {
                throw std::runtime_error(std::string("sf_close error: ") + sf_error_number(error));
            }
        }
	    SNDFILE *get() {
	        return f_;
	    }
	    SNDFILE *set(SNDFILE *f) {
            if (f_ == f) return f_;
            close();
	        f_ = f;
	        return f_;
	    }
	private:
	    SNDFILE *f_;
	};
}

#endif 