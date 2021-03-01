#include "bakuage/window_func.h"
#include "ipp.h"

namespace bakuage {
#if 0
    void CopyKeiserDouble(int n, double alpha, double *it, double scale) {
        ippsSet_64f(scale, it, n);
        ippsWinKaiser_64f_I(it, n, alpha);
    }
#endif
}
