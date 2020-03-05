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
    // Create Wavefunction
    hostPsi = (cuDoubleComplex *)malloc(sizeof(cuDoubleComplex) * DS);
    memcpy( hostPsi, arr, sizeof(cuDoubleComplex) * DS);
    // Allocate and initialize device data
    checkCudaErrors(cudaMalloc((void **)&devPsi, sizeof(cuDoubleComplex) * DS));
    checkCudaErrors(cudaMemcpy(devPsi, hostPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));

}

void Wavefunction::Step(double mult)
{
    multKernel(devPsi, mult, DIM, DIM);
}


void Wavefunction::MapColors(uchar4 *d_out)
{
    kernelLauncher(d_out, devPsi, DIM, DIM);
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