#ifndef CHAMBER_HPP
#define CHAMBER_HPP

#include <cstdlib>

// Handles the simulation parameters and potentials.
class Chamber
{
    public:
        int DIM, DS;
        double *Potential, *Kinetic, *XkY, *YkX, *devXkY, *devYkX, *omegaR;
        cuDoubleComplex *devExpPotential, *hostExpPotential;
        cuDoubleComplex *devExpKinetic, *hostExpKinetic;
        cuDoubleComplex *devExpXkY, *devExpYkX;
        double *X, *Y, *kX, *kY;
        double dx, dy, dk, dt;
        double fov, kfov;
        double Rxy, omegaZ;
        double omega, epsilon;
        double mass, a_s, a0, g;
        double cooling, useReal;
        double cmapscale;
        long atomNumber;
        bool useImaginaryTime, useRotatingFrame;
        cufftHandle fftPlan2D, fftPlan1D;
        
        void setup(int size, double g, double deltat, bool useImag, double cool);
        void setHarmonicPotential(double o, double ep);
        // void InitializePotential(cuDoubleComplex *arr);
        void Cleanup();
    

};




#endif