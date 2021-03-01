#ifndef BAKUAGE_WAIT_FREE_SINGLE_VALUE_QUEUE_H_
#define BAKUAGE_WAIT_FREE_SINGLE_VALUE_QUEUE_H_

#include <math.h>
#include <atomic>
#include <limits>

namespace bakuage {

template <class T>
class WaitFreeSingleValueQueue {
public:
    WaitFreeSingleValueQueue() {
        state_.store(0, std::memory_order_release);
    }
    void Push(const T &value) {
        static const uint8_t kBeginWriteStateTransition[16] = {4,0,6,7,0,0,0,0,25,0,0,7,29,0,29,0};
        static const uint8_t kEndWriteStateTransition[16] = {0,2,0,0,8,0,8,11,0,2,0,0,0,14,0,0};

        // begin write        
        int nextState = Transit(kBeginWriteStateTransition);

        // write
        int index = nextState / 16;
        values_[index] = value;

        // end write
        Transit(kEndWriteStateTransition);
    }
    bool TryPop(T *result) {
        static const uint8_t kBeginReadStateTransition[16] = {32,33,19,0,36,37,38,0,12,13,0,0,0,0,0,0};
        static const uint8_t kEndReadStateTransition[16] = {0,0,0,0,0,0,0,4,0,0,0,8,0,1,2,0};

        // begin read
        int nextState = TryTransit(kBeginReadStateTransition);
        if (nextState >= 16) {
            return false;
        }

        // read
        int index = nextState / 16;
        *result = values_[index];     
        
        // end read
        Transit(kEndReadStateTransition);

        return true;
    }
private:    
    int Transit(const uint8_t *transition) {        
        int nextState;
        while (1) {
			int currentState = state_.load(std::memory_order_relaxed);
            nextState = transition[currentState];
            if (state_.compare_exchange_weak(currentState, nextState % 16, 
                std::memory_order_release,
                std::memory_order_relaxed)) {
                break;
            }
        }
        return nextState;
    }
    int TryTransit(const uint8_t *transition) {
        int nextState;
        while (1) {
			int currentState = state_.load(std::memory_order_relaxed);
            nextState = transition[currentState];
            if (nextState >= 16) {
                return nextState;
            }
            if (state_.compare_exchange_weak(currentState, nextState % 16, 
                std::memory_order_release,
                std::memory_order_relaxed)) {
                break;
            }
        }
        return nextState;
    }

    std::atomic<int> state_;
    T values_[2];
};


// NaNをnon availableとして使う
template <class Float>
class WaitFreeFiniteFloatSingleValueQueue {
public:
    WaitFreeFiniteFloatSingleValueQueue() {
        value_.store(std::numeric_limits<Float>::quiet_NaN(), std::memory_order_release);
    }
    void Push(const Float &value) {
        value_.store(value, std::memory_order_release);
    }
    bool TryPop(Float *result) {
        if (_isnan(value_.load(std::memory_order_relaxed))) {
            return false;
        }
        Float v = value_.exchange(std::numeric_limits<Float>::quiet_NaN(), std::memory_order_release);
        if (_isnan(v)) {
            return false;
        }
        else {
            *result = v;
            return true;
        }
    }
private:    
    std::atomic<Float> value_;
};

}

#endif