#ifndef BAKUAGE_SPIN_LOCK_H_
#define BAKUAGE_SPIN_LOCK_H_

#include <atomic>

namespace bakuage {   

class SpinLock {
public:
	SpinLock() {
		Unlock();
	}
	void Lock() {
		while (!locked_.exchange(true, std::memory_order_release)) {}
	}
	void Unlock() {
		locked_.store(false, std::memory_order_release);
	}
private:
	std::atomic_bool locked_;
};

class ScopedLock {
public:
	ScopedLock(SpinLock *lock) : lock_(lock) {
		lock_->Lock();
	}
	~ScopedLock() {
		lock_->Unlock();
	}
private:
	SpinLock *lock_;
};

}

#endif 
