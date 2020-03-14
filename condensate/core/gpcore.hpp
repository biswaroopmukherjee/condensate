#ifndef GPCORE_HPP
#define GPCORE_HPP

#include "wavefunction.hpp"
#include "chamber.hpp"


void Setup(int size, double g, double deltat, bool useImag, double cool);
void SetHarmonicPotential(double omega, double epsilon);
void GetPotential(int vsizeX, int vsizeY, double *V);
void RotatingFrame(int size, double *omega_r);
void Evolve(int sizeX, int sizeY, cuDoubleComplex *arr, unsigned long steps, int skip, bool show);

namespace gpcore 
{
    extern Chamber chamber;
    extern Wavefunction Psi;
}

#endif