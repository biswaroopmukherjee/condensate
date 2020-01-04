#ifndef WAVEFUNCTION_HPP
#define WAVEFUNCTION_HPP

#include <cstdlib>

class Wavefunction
{
    private:
        cuDoubleComplex *devPsi, *hostPsi;
    public:
        void Initialize(cuDoubleComplex *arr);
        void Step(double mult);
        void MapColors(uchar4 *d_out);
        void ExportToVariable(cuDoubleComplex *arr);
        void Cleanup();

};

#endif