#ifndef BAKUAGE_LOOP_MACROS_H_
#define BAKUAGE_LOOP_MACROS_H_

// どっちがバイナリサイズ小さいか試したけど、大差なかった。

#if 1
#define _LOOP_UNFIXED_1(loop_variable, start, block) \
    { constexpr int loop_variable = (start); { block; } } 
#define _LOOP_UNFIXED_2(loop_variable, start, block) \
    _LOOP_UNFIXED_1(loop_variable, start, block) \
    _LOOP_UNFIXED_1(loop_variable, start + 1, block)
#define _LOOP_UNFIXED_4(loop_variable, start, block) \
    _LOOP_UNFIXED_2(loop_variable, start, block) \
    _LOOP_UNFIXED_2(loop_variable, start + 2, block)
#define _LOOP_UNFIXED_8(loop_variable, start, block) \
    _LOOP_UNFIXED_4(loop_variable, start, block) \
    _LOOP_UNFIXED_4(loop_variable, start + 4, block)
#define _LOOP_UNFIXED_16(loop_variable, start, block) \
    _LOOP_UNFIXED_8(loop_variable, start, block) \
    _LOOP_UNFIXED_8(loop_variable, start + 8, block)
#define _LOOP_UNFIXED_32(loop_variable, start, block) \
    _LOOP_UNFIXED_16(loop_variable, start, block) \
    _LOOP_UNFIXED_16(loop_variable, start + 16, block)

#define _LOOP_UNFIXED_N_FOR(loop_variable, start, n, block) \
    for (int loop_variable = (start); loop_variable < (start) + (n); loop_variable++) { \
        block; \
    }

#define _LOOP_UNFIXED_N_32(loop_variable, start, n, block) \
    if ((n) & 32) { \
        _LOOP_UNFIXED_32(loop_variable, start, block) \
    }

#define _LOOP_UNFIXED_N_16(loop_variable, start, n, block) \
    if ((n) & 16) { \
        _LOOP_UNFIXED_16(loop_variable, start, block) \
        _LOOP_UNFIXED_N_32(loop_variable, start + 16, n - 16, block) \
    } \
    else { \
        _LOOP_UNFIXED_N_32(loop_variable, start, n, block) \
    }

#define _LOOP_UNFIXED_N_8(loop_variable, start, n, block) \
    if ((n) & 8) { \
        _LOOP_UNFIXED_8(loop_variable, start, block) \
        _LOOP_UNFIXED_N_16(loop_variable, start + 8, n - 8, block) \
    } \
    else { \
        _LOOP_UNFIXED_N_16(loop_variable, start, n, block) \
    }

#define _LOOP_UNFIXED_N_4(loop_variable, start, n, block) \
    if ((n) & 4) { \
        _LOOP_UNFIXED_4(loop_variable, start, block) \
        _LOOP_UNFIXED_N_8(loop_variable, start + 4, n - 4, block) \
    } \
    else { \
        _LOOP_UNFIXED_N_8(loop_variable, start, n, block) \
    }

#define _LOOP_UNFIXED_N_2(loop_variable, start, n, block) \
    if ((n) & 2) { \
        _LOOP_UNFIXED_2(loop_variable, start, block) \
        _LOOP_UNFIXED_N_4(loop_variable, start + 2, n - 2, block) \
    } \
    else { \
        _LOOP_UNFIXED_N_4(loop_variable, start, n, block) \
    }

#define _LOOP_UNFIXED_N_1(loop_variable, start, n, block) \
    if ((n) & 1) { \
        _LOOP_UNFIXED_1(loop_variable, start, block) \
        _LOOP_UNFIXED_N_2(loop_variable, start + 1, n - 1, block) \
    } \
    else { \
        _LOOP_UNFIXED_N_2(loop_variable, start, n, block) \
    }

#define LOOP_RANGE(loop_variable, start, end, block) { \
    constexpr int _start_fixed = (start); \
    constexpr int _n_fixed = (end) - _start_fixed; \
    _LOOP_UNFIXED_N_1(loop_variable, _start_fixed, _n_fixed, block) \
}
#else

#define _LOOP_RANGE_IMPL(loop_variable, start, end, block, i) \
	if (i < (end) - (start)) { constexpr int loop_variable = (start) + i; block; }

#define LOOP_RANGE(loop_variable, start, end, block) { \
	static_assert((end) - (start) <= 32, "loop count have to be <= 32"); \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 0) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 1) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 2) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 3) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 4) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 5) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 6) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 7) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 8) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 9) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 10) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 11) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 12) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 13) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 14) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 15) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 16) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 17) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 18) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 19) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 20) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 22) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 23) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 24) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 25) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 26) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 27) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 28) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 29) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 30) \
    _LOOP_RANGE_IMPL(loop_variable, start, end, block, 31) \
}

#endif

#endif
