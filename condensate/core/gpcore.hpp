#ifndef GPCORE_HPP
#define GPCORE_HPP

#include "wavefunction.hpp"
#include "chamber.hpp"


void Setup(int size, double fov, double g, double deltat, bool useImag, double cool);
void SetHarmonicPotential(double omega, double epsilon);
void SetEdgePotential(double strength, double radius, double sharpness);
void GetPotential(int vsizeX, int vsizeY, double *V);
void SetPotential(int vsizeX, int vsizeY, double *V);
void RotatingFrame(int size, double *omega_r);
void AbsorbingBoundaryConditions(double strength, double radius);
void SetupSpoon(double strength, double radius);
void SetupLeapMotion(double centerx, double centery, double zoomx, double zoomy, bool controlstrength);
void Evolve(int sizeX, int sizeY, cuDoubleComplex *arr, 
            unsigned long steps, int skip, bool show, double vmax);

namespace gpcore 
{
    extern Chamber chamber;
    extern Wavefunction Psi;
}

#endif