#ifndef BAKUAGE_BAKUAGE_BESSEL_H_
#define BAKUAGE_BAKUAGE_BESSEL_H_

#include <cmath>

double dbesi0(double x);

namespace bakuage {

    double BesselI0(double x) {
        return dbesi0(x);
    }

}

#endif
