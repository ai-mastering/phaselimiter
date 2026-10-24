#ifndef BAKUAGE_DSP_H
#define BAKUAGE_DSP_H

#include <iostream>
#ifndef BAKUAGE_USE_ACCELERATE
#include <ipp.h>
#endif

namespace bakuage {
inline void InitializeDsp() {
#ifdef BAKUAGE_USE_ACCELERATE
    std::cerr << "DSP backend: Apple Accelerate" << std::endl;
#else
    ippInit();
    const IppLibraryVersion *lib = ippGetLibVersion();
    std::cerr << "Ipp initialized " << lib->Name << " " << lib->Version << std::endl;
#endif
}
}
#endif
