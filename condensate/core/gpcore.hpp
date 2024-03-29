#ifndef GPCORE_HPP
#define GPCORE_HPP

#include "wavefunction.hpp"
#include "chamber.hpp"


void Setup(int size, double fov, int size_g, double *g, double deltat, bool useImag, double cool, bool every_time_reset_potential);
void SetHarmonicPotential(int size_o, double *omega_t, int size_e, double *epsilon_t);
void SetEdgePotential(double strength, double radius, double sharpness);
void GetPotential(int vsizeX, int vsizeY, double *V);
void SetPotential(int vsizeX, int vsizeY, double *V);
void RotatingFrame(int size, double *omega_r);
void AbsorbingBoundaryConditions(double strength, double radius);
void SetupSpoon(double strength, double radius);
void SetupLeapMotion(double centerx, double centery, double zoomx, double zoomy, bool controlstrength);
void Evolve(int sizeX, int sizeY, cuDoubleComplex *arr, 
            unsigned long steps, int skip, bool show, double vmax, char *filename);

namespace gpcore 
{
    extern Chamber chamber;
    extern Wavefunction Psi;
}

#endif