#pragma once

#include "bakuage/concurrentqueue.h"

namespace bakuage {

template <class T, int S>
class FixedSizeMemoryPool {
public:
	FixedSizeMemoryPool() {}

	T *Alloc() {

	}

	void Dealloc() {

	}
private:
	moodycamel::ConcurrentQueue<T *> queue_;
	std::array<T, S> buffer_;
};

template <class T>
class LockFreeAllocator {
public:
	typedef T value_type;
	typedef T *pointer;
	typedef size_t size_type;

	pointer allocate(size_type n) {
		return nullptr;
	}
	void deallocate(pointer p, size_type n) {

	}
};

}