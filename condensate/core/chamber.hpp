#ifndef CHAMBER_HPP
#define CHAMBER_HPP

#include <cstdlib>

// Settings for spoons
struct spoonProps {
    double strengthSetting;
    double strength; 
    int radius;
    int2 pos;
};


// Handles the simulation parameters and potentials.
class Chamber
{
    public:
        int DIM, DS;
        double *Potential, *Kinetic, *XkY, *YkX, *XX, *YY, *devXkY, *devYkX, *devXX, *devYY, *omegaR;
        double *devPotential;
        cuDoubleComplex *devExpPotential, *hostExpPotential;
        cuDoubleComplex *devExpKinetic, *hostExpKinetic;
        cuDoubleComplex *devExpXkY, *devExpYkX;
        double *X, *Y, *kX, *kY;
        double dx, dy, dk, dt;
        double fov, kfov;
        double *omega, *epsilon, *g;
        double mass;
        double cooling, useReal;
        double cmapscale;
        bool useImaginaryTime, useRotatingFrame;
        bool stopSim;
        spoonProps spoon1;
        bool useLeapMotion, useLeapZ;
        double4 LeapProps;
        cufftHandle fftPlan2D, fftPlan1D;
        bool every_time_reset_potential;


        
        void setup(int size, double fov, int size_g, double *g, double deltat, bool useImag, double cool, bool every_time_reset_potential);
        void setHarmonicPotential(double o, double ep);
        void differencePotential(double o1, double o2, double ep1, double ep2);
        void setEdgePotential(double strength, double radius, double sharpness);
        void setupPotential();
        void AbsorbingBoundaryConditions(double strength, double radius);
        void SetupSpoon(double strength, double radius);
        void Spoon();
        void TimeVary(int step_idx);
        // void InitializePotential(cuDoubleComplex *arr);
        void Cleanup();
    

};




#endif