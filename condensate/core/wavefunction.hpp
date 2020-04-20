#ifndef WAVEFUNCTION_HPP
#define WAVEFUNCTION_HPP

#include <cstdlib>

class Wavefunction
{
    private:
        cuDoubleComplex *devPsi, *hostPsi;
        double *devDensity;
    public:
        int DIM;
        void Initialize( cuDoubleComplex *arr);
        void Step( double mult);
        void RealSpaceHalfStep();
        void MomentumSpaceStep();
        void Renormalize();
        void RotatingFrame(unsigned long timestep, unsigned long steps);
        void MapColors(uchar4 *d_out);
        void CalculateEnergy(double energy);
        
        void ExportToVariable(cuDoubleComplex *arr);
        void Cleanup();

};

#endif