// NumPy 1.14.3 double argpartition/argsort, specialized from npysort.
// Preserve unstable tie handling used by scikit-learn 0.19.1 brute neighbors.
// See NUMPY_LICENSE.txt. Original sort code by Charles R. Harris.
#include "classic_detector/numpy_neighbors.h"
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
namespace classic_detector { namespace {
using npy_intp = std::ptrdiff_t;
#define NPY_INLINE inline
#define NPY_MAX_PIVOT_STACK 50
#define DOUBLE_LT(a,b) ((a) < (b) || (std::isnan(b) && !std::isnan(a)))
#define INTP_SWAP(a,b) { std::swap(a,b); }
#define DOUBLE_SWAP(a,b) { std::swap(a,b); }
#define NOT_USED unused
#define NPY_UNLIKELY(x) (x)
#define PYA_QS_STACK (sizeof(npy_intp) * 16)
#define SMALL_QUICKSORT 15
int npy_get_msb(npy_intp n) { int k = 0; while (n >>= 1) ++k; return k; }
int aintroselect_double(double *, npy_intp *, npy_intp, npy_intp, npy_intp *, npy_intp *, void *);
int
aheapsort_double(void *vv, npy_intp *tosort, npy_intp n, void *NOT_USED)
{
    double *v = static_cast<double *>(vv);
    npy_intp *a, i,j,l, tmp;
    /* One-based heap indices, translated at each array access. */
    a = tosort;

    for (l = n>>1; l > 0; --l) {
        tmp = a[(l) - 1];
        for (i = l, j = l<<1; j <= n;) {
            if (j < n && DOUBLE_LT(v[a[(j) - 1]], v[a[(j+1) - 1]])) {
                j += 1;
            }
            if (DOUBLE_LT(v[tmp], v[a[(j) - 1]])) {
                a[(i) - 1] = a[(j) - 1];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[(i) - 1] = tmp;
    }

    for (; n > 1;) {
        tmp = a[(n) - 1];
        a[(n) - 1] = a[(1) - 1];
        n -= 1;
        for (i = 1, j = 2; j <= n;) {
            if (j < n && DOUBLE_LT(v[a[(j) - 1]], v[a[(j+1) - 1]])) {
                j++;
            }
            if (DOUBLE_LT(v[tmp], v[a[(j) - 1]])) {
                a[(i) - 1] = a[(j) - 1];
                i = j;
                j += j;
            }
            else {
                break;
            }
        }
        a[(i) - 1] = tmp;
    }

    return 0;
}

static NPY_INLINE void store_pivot(npy_intp pivot, npy_intp kth,
                                   npy_intp * pivots, npy_intp * npiv)
{
    if (pivots == NULL) {
        return;
    }

    /*
     * If pivot is the requested kth store it, overwritting other pivots if
     * required. This must be done so iterative partition can work without
     * manually shifting lower data offset by kth each time
     */
    if (pivot == kth && *npiv == NPY_MAX_PIVOT_STACK) {
        pivots[*npiv - 1] = pivot;
    }
    /*
     * we only need pivots larger than current kth, larger pivots are not
     * useful as partitions on smaller kth would reorder the stored pivots
     */
    else if (pivot >= kth && *npiv < NPY_MAX_PIVOT_STACK) {
        pivots[*npiv] = pivot;
        (*npiv) += 1;
    }
}


static npy_intp
amedian_of_median5_double(double *v, npy_intp* tosort, const npy_intp num,
                         npy_intp * pivots,
                         npy_intp * npiv);



/* helper macros to avoid duplication of direct/indirect selection */
#define IDX(x) tosort[x]
#define SORTEE(x) tosort[x]
#define SWAP INTP_SWAP
#define MEDIAN3_SWAP(v, tosort, low, mid, high) \
    amedian3_swap_double(v, tosort, low, mid, high)
#define MEDIAN5(v, tosort, subleft) \
        amedian5_double(v, tosort + subleft)
#define UNGUARDED_PARTITION(v, tosort, pivot, ll, hh) \
        aunguarded_partition_double(v, tosort, pivot, ll, hh)
#define INTROSELECT(v, tosort, num, kth, pivots, npiv) \
        aintroselect_double(v, tosort, nmed, nmed / 2, pivots, npiv, NULL)
#define DUMBSELECT(v, tosort, left, num, kth) \
        adumb_select_double(v, tosort + left, num, kth)


/*
 * median of 3 pivot strategy
 * gets min and median and moves median to low and min to low + 1
 * for efficient partitioning, see unguarded_partition
 */
static NPY_INLINE void
amedian3_swap_double(double * v,
                          npy_intp * tosort,
                          npy_intp low, npy_intp mid, npy_intp high)
{
    if (DOUBLE_LT(v[IDX(high)], v[IDX(mid)]))
        SWAP(SORTEE(high), SORTEE(mid));
    if (DOUBLE_LT(v[IDX(high)], v[IDX(low)]))
        SWAP(SORTEE(high), SORTEE(low));
    /* move pivot to low */
    if (DOUBLE_LT(v[IDX(low)], v[IDX(mid)]))
        SWAP(SORTEE(low), SORTEE(mid));
    /* move 3-lowest element to low + 1 */
    SWAP(SORTEE(mid), SORTEE(low + 1));
}


/* select index of median of five elements */
static npy_intp amedian5_double(
                                    const double * v, npy_intp * tosort
                                    )
{
    /* could be optimized as we only need the index (no swaps) */
    if (DOUBLE_LT(v[IDX(1)], v[IDX(0)])) {
        SWAP(SORTEE(1), SORTEE(0));
    }
    if (DOUBLE_LT(v[IDX(4)], v[IDX(3)])) {
        SWAP(SORTEE(4), SORTEE(3));
    }
    if (DOUBLE_LT(v[IDX(3)], v[IDX(0)])) {
        SWAP(SORTEE(3), SORTEE(0));
    }
    if (DOUBLE_LT(v[IDX(4)], v[IDX(1)])) {
        SWAP(SORTEE(4), SORTEE(1));
    }
    if (DOUBLE_LT(v[IDX(2)], v[IDX(1)])) {
        SWAP(SORTEE(2), SORTEE(1));
    }
    if (DOUBLE_LT(v[IDX(3)], v[IDX(2)])) {
        if (DOUBLE_LT(v[IDX(3)], v[IDX(1)])) {
            return 1;
        }
        else {
            return 3;
        }
    }
    else {
        /* v[1] and v[2] swapped into order above */
        return 2;
    }
}


/*
 * partition and return the index were the pivot belongs
 * the data must have following property to avoid bound checks:
 *                  ll ... hh
 * lower-than-pivot [x x x x] larger-than-pivot
 */
static NPY_INLINE void
aunguarded_partition_double(double * v,
                                 npy_intp * tosort,
                                 const double pivot,
                                 npy_intp * ll, npy_intp * hh)
{
    for (;;) {
        do (*ll)++; while (DOUBLE_LT(v[IDX(*ll)], pivot));
        do (*hh)--; while (DOUBLE_LT(pivot, v[IDX(*hh)]));

        if (*hh < *ll)
            break;

        SWAP(SORTEE(*ll), SORTEE(*hh));
    }
}


/*
 * select median of median of blocks of 5
 * if used as partition pivot it splits the range into at least 30%/70%
 * allowing linear time worstcase quickselect
 */
static npy_intp
amedian_of_median5_double(double *v,
                               npy_intp* tosort,
                               const npy_intp num,
                               npy_intp * pivots,
                               npy_intp * npiv)
{
    npy_intp i, subleft;
    npy_intp right = num - 1;
    npy_intp nmed = (right + 1) / 5;
    for (i = 0, subleft = 0; i < nmed; i++, subleft += 5) {
        npy_intp m = MEDIAN5(v, tosort, subleft);
        SWAP(SORTEE(subleft + m), SORTEE(i));
    }

    if (nmed > 2)
        INTROSELECT(v, tosort, nmed, nmed / 2, pivots, npiv);
    return nmed / 2;
}


/*
 * N^2 selection, fast only for very small kth
 * useful for close multiple partitions
 * (e.g. even element median, interpolating percentile)
 */
static int
adumb_select_double(double *v,
                         npy_intp * tosort,
                         npy_intp num, npy_intp kth)
{
    npy_intp i;
    for (i = 0; i <= kth; i++) {
        npy_intp minidx = i;
        double minval = v[IDX(i)];
        npy_intp k;
        for (k = i + 1; k < num; k++) {
            if (DOUBLE_LT(v[IDX(k)], minval)) {
                minidx = k;
                minval = v[IDX(k)];
            }
        }
        SWAP(SORTEE(i), SORTEE(minidx));
    }

    return 0;
}


/*
 * iterative median of 3 quickselect with cutoff to median-of-medians-of5
 * receives stack of already computed pivots in v to minimize the
 * partition size were kth is searched in
 *
 * area that needs partitioning in [...]
 * kth 0:  [8  7  6  5  4  3  2  1  0] -> med3 partitions elements [4, 2, 0]
 *          0  1  2  3  4  8  7  5  6  -> pop requested kth -> stack [4, 2]
 * kth 3:   0  1  2 [3] 4  8  7  5  6  -> stack [4]
 * kth 5:   0  1  2  3  4 [8  7  5  6] -> stack [6]
 * kth 8:   0  1  2  3  4  5  6 [8  7] -> stack []
 *
 */
int
aintroselect_double(double *v,
                         npy_intp* tosort,
                         npy_intp num, npy_intp kth,
                         npy_intp * pivots,
                         npy_intp * npiv,
                         void *NOT_USED)
{
    npy_intp low  = 0;
    npy_intp high = num - 1;
    int depth_limit;

    if (npiv == NULL)
        pivots = NULL;

    while (pivots != NULL && *npiv > 0) {
        if (pivots[*npiv - 1] > kth) {
            /* pivot larger than kth set it as upper bound */
            high = pivots[*npiv - 1] - 1;
            break;
        }
        else if (pivots[*npiv - 1] == kth) {
            /* kth was already found in a previous iteration -> done */
            return 0;
        }

        low = pivots[*npiv - 1] + 1;

        /* pop from stack */
        *npiv -= 1;
    }

    /*
     * use a faster O(n*kth) algorithm for very small kth
     * e.g. for interpolating percentile
     */
    if (kth - low < 3) {
        DUMBSELECT(v, tosort, low, high - low + 1, kth - low);
        store_pivot(kth, kth, pivots, npiv);
        return 0;
    }
    else if (1 && kth == num - 1) {
        /* useful to check if NaN present via partition(d, (x, -1)) */
        npy_intp k;
        npy_intp maxidx = low;
        double maxval = v[IDX(low)];
        for (k = low + 1; k < num; k++) {
            if (!DOUBLE_LT(v[IDX(k)], maxval)) {
                maxidx = k;
                maxval = v[IDX(k)];
            }
        }
        SWAP(SORTEE(kth), SORTEE(maxidx));
        return 0;
    }

    depth_limit = npy_get_msb(num) * 2;

    /* guarantee three elements */
    for (;low + 1 < high;) {
        npy_intp       ll = low + 1;
        npy_intp       hh = high;

        /*
         * if we aren't making sufficient progress with median of 3
         * fall back to median-of-median5 pivot for linear worst case
         * med3 for small sizes is required to do unguarded partition
         */
        if (depth_limit > 0 || hh - ll < 5) {
            const npy_intp mid = low + (high - low) / 2;
            /* median of 3 pivot strategy,
             * swapping for efficient partition */
            MEDIAN3_SWAP(v, tosort, low, mid, high);
        }
        else {
            npy_intp mid;
            /* FIXME: always use pivots to optimize this iterative partition */
            mid = ll + amedian_of_median5_double(v, tosort + ll, hh - ll, NULL, NULL);
            SWAP(SORTEE(mid), SORTEE(low));
            /* adapt for the larger partition than med3 pivot */
            ll--;
            hh++;
        }

        depth_limit--;

        /*
         * find place to put pivot (in low):
         * previous swapping removes need for bound checks
         * pivot 3-lowest [x x x] 3-highest
         */
        UNGUARDED_PARTITION(v, tosort, v[IDX(low)], &ll, &hh);

        /* move pivot into position */
        SWAP(SORTEE(low), SORTEE(hh));

        /* kth pivot stored later */
        if (hh != kth) {
            store_pivot(hh, kth, pivots, npiv);
        }

        if (hh >= kth)
            high = hh - 1;
        if (hh <= kth)
            low = ll;
    }

    /* two elements */
    if (high == low + 1) {
        if (DOUBLE_LT(v[IDX(high)], v[IDX(low)])) {
            SWAP(SORTEE(high), SORTEE(low))
        }
    }
    store_pivot(kth, kth, pivots, npiv);

    return 0;
}


#undef IDX
#undef SWAP
#undef SORTEE
#undef MEDIAN3_SWAP
#undef MEDIAN5
#undef UNGUARDED_PARTITION
#undef INTROSELECT
#undef DUMBSELECT
int
aquicksort_double(void *vv, npy_intp* tosort, npy_intp num, void *NOT_USED)
{
    double *v = static_cast<double *>(vv);
    double vp;
    npy_intp *pl = tosort;
    npy_intp *pr = tosort + num - 1;
    npy_intp *stack[PYA_QS_STACK];
    npy_intp **sptr = stack;
    npy_intp *pm, *pi, *pj, *pk, vi;
    int depth[PYA_QS_STACK];
    int * psdepth = depth;
    int cdepth = npy_get_msb(num) * 2;

    for (;;) {
        if (NPY_UNLIKELY(cdepth < 0)) {
            aheapsort_double(vv, pl, pr - pl + 1, NULL);
            goto stack_pop;
        }
        while ((pr - pl) > SMALL_QUICKSORT) {
            /* quicksort partition */
            pm = pl + ((pr - pl) >> 1);
            if (DOUBLE_LT(v[*pm],v[*pl])) INTP_SWAP(*pm, *pl);
            if (DOUBLE_LT(v[*pr],v[*pm])) INTP_SWAP(*pr, *pm);
            if (DOUBLE_LT(v[*pm],v[*pl])) INTP_SWAP(*pm, *pl);
            vp = v[*pm];
            pi = pl;
            pj = pr - 1;
            INTP_SWAP(*pm, *pj);
            for (;;) {
                do ++pi; while (DOUBLE_LT(v[*pi], vp));
                do --pj; while (DOUBLE_LT(vp, v[*pj]));
                if (pi >= pj) {
                    break;
                }
                INTP_SWAP(*pi, *pj);
            }
            pk = pr - 1;
            INTP_SWAP(*pi,*pk);
            /* push largest partition on stack */
            if (pi - pl < pr - pi) {
                *sptr++ = pi + 1;
                *sptr++ = pr;
                pr = pi - 1;
            }
            else {
                *sptr++ = pl;
                *sptr++ = pi - 1;
                pl = pi + 1;
            }
            *psdepth++ = --cdepth;
        }

        /* insertion sort */
        for (pi = pl + 1; pi <= pr; ++pi) {
            vi = *pi;
            vp = v[vi];
            pj = pi;
            pk = pi - 1;
            while (pj > pl && DOUBLE_LT(vp, v[*pk])) {
                *pj-- = *pk--;
            }
            *pj = vi;
        }
stack_pop:
        if (sptr == stack) {
            break;
        }
        pr = *(--sptr);
        pl = *(--sptr);
        cdepth = *(--psdepth);
    }

    return 0;
}


}  // namespace
std::vector<size_t> NumpyNeighbors(const std::vector<double> &distances, size_t count) {
    if (count == 0 || count > distances.size()) throw std::invalid_argument("invalid neighbor count");
    std::vector<double> values(distances);
    std::vector<npy_intp> indices(values.size());
    std::iota(indices.begin(), indices.end(), 0);
    aintroselect_double(values.data(), indices.data(), indices.size(), count - 1, nullptr, nullptr, nullptr);
    indices.resize(count);
    std::vector<double> selected(count);
    for (size_t i = 0; i < count; ++i) selected[i] = values[indices[i]];
    std::vector<npy_intp> order(count);
    std::iota(order.begin(), order.end(), 0);
    aquicksort_double(selected.data(), order.data(), count, nullptr);
    std::vector<size_t> result(count);
    for (size_t i = 0; i < count; ++i) result[i] = indices[order[i]];
    return result;
}
}  // namespace classic_detector
