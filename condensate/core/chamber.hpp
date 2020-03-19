#ifndef CHAMBER_HPP
#define CHAMBER_HPP

#include <cstdlib>

// Handles the simulation parameters and potentials.
class Chamber
{
    public:
        int DIM, DS;
        double *Potential, *Kinetic, *XkY, *YkX, *devXkY, *devYkX, *omegaR;
        // double *devPotential;
        cuDoubleComplex *devExpPotential, *hostExpPotential;
        cuDoubleComplex *devExpKinetic, *hostExpKinetic;
        cuDoubleComplex *devExpXkY, *devExpYkX;
        double *X, *Y, *kX, *kY;
        double dx, dy, dk, dt;
        double fov, kfov;
        double omega, epsilon;
        double mass, g;
        double cooling, useReal;
        double cmapscale;
        bool useImaginaryTime, useRotatingFrame;
        cufftHandle fftPlan2D, fftPlan1D;
        
        void setup(int size, double fov, double g, double deltat, bool useImag, double cool);
        void setHarmonicPotential(double o, double ep);
        void AbsorbingBoundaryConditions(double strength, double radius);
        // void InitializePotential(cuDoubleComplex *arr);
        void Cleanup();
    

};




#endif