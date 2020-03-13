#ifndef GPCORE_HPP
#define GPCORE_HPP

#include "wavefunction.hpp"
#include "chamber.hpp"


void setup(int size, double deltat, bool useImag, double cool);
void setHarmonicPotential(double omega, double epsilon);
void getPotential(int vsizeX, int vsizeY, double *V);
void Evolve(int sizeX, int sizeY, cuDoubleComplex *arr, long steps, int skip, bool show);

namespace gpcore 
{
    extern Chamber chamber;
    extern Wavefunction Psi;
}

#endif