#ifndef INITIALIZE_HPP
#define INITIALIZE_HPP

#include <cstdlib>

namespace gpcore {
    extern cuDoubleComplex *devPsi, *hostV, *hostPsi;
    extern int wWidth, wHeight;
    void initialize();
    void initializecpy(cuDoubleComplex *arr);
}

#endif