#ifndef GPCORE_HPP
#define GPCORE_HPP

#include "wavefunction.hpp"

void evolve(int sizex, int sizey, cuDoubleComplex *arr);

namespace gpcore {
    extern Wavefunction Psi;
    extern int wWidth, wHeight ;
}

#endif