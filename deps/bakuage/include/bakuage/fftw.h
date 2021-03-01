#ifndef BAKUAGE_FFTW_H_
#define BAKUAGE_FFTW_H_

#include <mutex>

namespace bakuage {
class FFTW {
public:
	static std::recursive_mutex &mutex() {
		static std::recursive_mutex mutex_;
		return mutex_;
	}
};
}

#endif