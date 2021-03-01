#ifndef BAKUAGE_BAKUAGE_SNDFILE_WRAPPER_H_
#define BAKUAGE_BAKUAGE_SNDFILE_WRAPPER_H_

#include "sndfile.h"

namespace bakuage {
	class SndfileWrapper {
	public:
	    SndfileWrapper(): f_(NULL) {}
	    virtual ~SndfileWrapper() {
	        set(NULL);
	    }
	    SNDFILE *get() {
	        return f_;
	    }
	    SNDFILE *set(SNDFILE *f) {
	        if (f_) {
	            sf_close(f_);
	        }
	        f_ = f;
	        return f_;
	    }
	private:
	    SNDFILE *f_;
	};
}

#endif 