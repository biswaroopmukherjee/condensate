#ifndef WAVEFUNCTION_HPP
#define WAVEFUNCTION_HPP

#include <cstdlib>

class Wavefunction
{
    private:
        cuDoubleComplex *devPsi, *hostPsi;
        double *devDensity;
        int *devMovieBuffer, *hostMovieBuffer;
        FILE *ffmpeg;
    public:
        int DIM;
        void Initialize( cuDoubleComplex *arr);
        void InitializeMovie(char *filename);
        void Step( double mult);
        void RealSpaceHalfStep(int timestep);
        void MomentumSpaceStep();
        void Renormalize();
        void RotatingFrame(unsigned long timestep, unsigned long steps);
        void MapColors(uchar4 *d_out);
        void MovieFrame();
        void CalculateEnergy(double energy);
        
        void ExportToVariable(cuDoubleComplex *arr);
        void MovieCleanup();
        void Cleanup();

};

#endif