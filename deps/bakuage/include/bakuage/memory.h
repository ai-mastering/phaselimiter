#ifndef BAKUAGE_MEMORY_H_
#define BAKUAGE_MEMORY_H_

#include <cstring>
#include <assert.h>
#include <stdint.h>
#include <algorithm>
#include <boost/serialization/split_member.hpp>
#include "bakuage/utils.h"

namespace boost {
    namespace serialization {
        class access;
    }
}

namespace bakuage {

	// 長さもアラインするので、 + 2 * alignment
    void *AlignedMalloc(size_t size, size_t alignment = 64);
    void AlignedFree(void *ptr);
    
	template <class T>
	class AlignedPodVector {
	public:
		static_assert(IsTrivial<T>::value, "AlignedPodVector T must be trivial");

		AlignedPodVector(): size_(0), allocated_size_(0), data_(nullptr) {}
		AlignedPodVector(size_t len): size_(0), allocated_size_(0), data_(nullptr) {
			resize(len);
		}
		AlignedPodVector(size_t len, const T &value) : size_(0), allocated_size_(0), data_(nullptr) {
			resize(len);
			for (int i = 0; i < len; i++) {
				data_[i] = value;
			}
		}
        template <class It>
        AlignedPodVector(It bg, It ed) : size_(0), allocated_size_(0), data_(nullptr) {
            resize(std::distance(bg, ed));
            int i = 0;
            for (auto it = bg; it != ed; ++it) {
                data_[i] = *it;
                i++;
            }
        }
        template <class S>
        AlignedPodVector(std::initializer_list<S> list): AlignedPodVector(list.begin(), list.end()) {}

		AlignedPodVector(const AlignedPodVector& x): size_(0), allocated_size_(0), data_(nullptr) {
			resize(x.size());
            if (size_) {
                TypedMemcpy(data_, x.data_, size_);
            }
		}
		AlignedPodVector(AlignedPodVector&& x): size_(x.size_), allocated_size_(x.allocated_size_), data_(x.data_) {
			x.data_ = nullptr;
			x.size_ = 0;
            x.allocated_size_ = 0;
		}
        
        ~AlignedPodVector() {
            if (data_) AlignedFree(data_);
        }

		AlignedPodVector& operator=(const AlignedPodVector& x) {
			resize(x.size());
            if (size_) {
                TypedMemcpy(data_, x.data_, size_);
            }
			return *this;
		}
		AlignedPodVector& operator=(AlignedPodVector&& x) {
			if (data_) AlignedFree(data_);
			data_ = x.data_;
			size_ = x.size_;
            allocated_size_ = x.allocated_size_;
			x.data_ = nullptr;
			x.size_ = 0;
            x.allocated_size_ = 0;
			return *this;
		}
		AlignedPodVector& operator=(std::initializer_list<T> x) {
			resize(x.size());
			int i = 0;
			for (auto it = x.begin(); it != x.end(); ++it) {
				x[i] = *it;
				i++;
			}
			return *this;
		}

		bool operator==(const AlignedPodVector& x) const {
			return std::equal(data_, data_ + size_, x.data_, x.data_ + x.size_);
		}

        void push_back(const T& x) {
            resize(size_ + 1);
            data_[size_ - 1] = x;
        }
        
		T *data() { return data_; }
		const T *data() const { return data_; }
		T *begin() { return data_; }
        const T *begin() const { return data_; }
		T *end() { return data_ + size_; }
        const T *end() const { return data_ + size_; }
		const T *cbegin() const { return data_; }
		const T *cend() const { return data_ + size_; }
		size_t size() const { return size_; }
		T &operator[](size_t i) { return data_[i]; }
		const T &operator[](size_t i) const { return data_[i]; }
		void resize(size_t len) {
			if (len > allocated_size_) {
                DoRealloc(std::max<size_t>(len, 2 * allocated_size_));
			}
			size_ = len;
		}
        void shrink_to_fit() {
            DoRealloc(size_);
        }
	private:
        friend class boost::serialization::access;
        template<class Archive>
        void save(Archive & ar, const unsigned int version) const {
            ar << size_;
            for (size_t i = 0; i < size_; i++) {
                ar << data_[i];
            }
        }
        template<class Archive>
        void load(Archive & ar, const unsigned int version) {
            size_t size;
            ar >> size;
            resize(size);
            for (size_t i = 0; i < size; i++) {
                ar >> data_[i];
            }
        }
        BOOST_SERIALIZATION_SPLIT_MEMBER()
        
        void DoRealloc(size_t new_allocated_size) {
            if (allocated_size_ == new_allocated_size) return;
            allocated_size_ = new_allocated_size;
            if (allocated_size_) {
                if (data_) {
                    T *new_data = (T *)AlignedMalloc(sizeof(T) * allocated_size_);
                    TypedMemcpy(new_data, data_, size_);
                    TypedFillZero(new_data + size_, allocated_size_ - size_);
                    AlignedFree(data_);
                    data_ = new_data;
                }
                else {
                    data_ = (T *)AlignedMalloc(sizeof(T) * allocated_size_);
                    TypedFillZero(data_, allocated_size_);
                }
            } else {
                if (data_) {
                    AlignedFree(data_);
                    data_ = nullptr;
                }
            }
        }
		size_t size_;
        size_t allocated_size_;
		T *data_;
	};

    size_t GetPeakRss();
    size_t GetCurrentRss();
}

#ifdef _MSC_VER
#define BAKUAGE_ALIGN(x) __declspec(align(x))
#define BAKUAGE_TLS __declspec(thread)
#else
#define BAKUAGE_ALIGN(x) __attribute__((aligned(x)))
#define BAKUAGE_TLS __thread
#endif

#endif 
