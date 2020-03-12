#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>  
#include <algorithm>

#include "gpcore.hpp"
#include "helper_cudagl.h"
#include "wavefunction.hpp"
#include "gp_kernels.h"


void Wavefunction::Initialize(cuDoubleComplex *arr)
{
    DIM = gpcore::chamber.DIM;
    int DS = DIM * DIM;
    int cudoubleDS = sizeof(cuDoubleComplex) * DS;
    // Create Wavefunction
    hostPsi = (cuDoubleComplex *)malloc(cudoubleDS);
    memcpy( hostPsi, arr, cudoubleDS);
    // Allocate and initialize device data
    checkCudaErrors(cudaMalloc((void **)&devPsi, cudoubleDS));
    checkCudaErrors(cudaMalloc((void **)&devDensity, sizeof(double) * DS));
    checkCudaErrors(cudaMemcpy(devPsi, hostPsi, cudoubleDS, cudaMemcpyHostToDevice));
}

void Wavefunction::Step(double mult)
{
    multKernelLauncher(devPsi, mult, DIM, DIM);
}


void Wavefunction::MapColors(uchar4 *d_out)
{
    colormapKernelLauncher(d_out, devPsi, DIM, DIM);
}

void Wavefunction::RealSpaceHalfStep() {
    realspaceKernelLauncher(
                    devPsi, 
                    gpcore::chamber.devExpPotential, 
                    devPsi, 
                    gpcore::chamber.dt,
                    gpcore::chamber.useReal,
                    gpcore::chamber.cooling,
                    DIM, DIM);
}

void Wavefunction::MomentumSpaceStep() {
    cufftExecZ2Z(gpcore::chamber.fftPlan2D, devPsi, devPsi, CUFFT_FORWARD);   
    multKernelLauncher(devPsi, 1.0/DIM, DIM, DIM); // fft renorm
    momentumspaceKernelLauncher(devPsi, gpcore::chamber.devExpKinetic, devPsi, DIM, DIM);
    cufftExecZ2Z(gpcore::chamber.fftPlan2D, devPsi, devPsi, CUFFT_INVERSE);  
    multKernelLauncher(devPsi, 1.0/DIM, DIM, DIM); // fft renorm
}

void Wavefunction::Renormalize() {
    parSum(devPsi, devDensity, gpcore::chamber.dx, DIM, DIM);
}


void Wavefunction::ExportToVariable(cuDoubleComplex *arr)
{
    int DS = DIM * DIM;
    checkCudaErrors(cudaMemcpy(hostPsi, devPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyDeviceToHost));
    memcpy(arr, hostPsi, sizeof(cuDoubleComplex) * DS);
}


void Wavefunction::Cleanup()
{
    free(hostPsi);
    checkCudaErrors(cudaFree(devPsi));
    checkCudaErrors(cudaDeviceReset());
}