#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <cstring>

using namespace std;

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

void Wavefunction::InitializeMovie(char *filename)
{
    int DS = DIM * DIM;
    // Create and allocate buffers on the host and the device for movie creation
    hostMovieBuffer = (int *)malloc(sizeof(int) * DS);
    checkCudaErrors(cudaMalloc((void **)&devMovieBuffer, sizeof(int)*DS));
    // Generate the ffmpeg command
    string sDIM = to_string(DIM);
    string cmd = "ffmpeg -r 30 -f rawvideo -pix_fmt rgba -s " + sDIM + "x" + sDIM + " -i - "
                  "-threads 0 -preset fast -y -pix_fmt yuv420p -crf 15 " + filename;
    const char *ccmd = cmd.c_str();

    ffmpeg = popen(ccmd, "w");
}

void Wavefunction::Step(double mult)
{
    multKernelLauncher(devPsi, mult, DIM, DIM);
}


void Wavefunction::MapColors(uchar4 *d_out)
{
    colormapKernelLauncher(d_out, devPsi, gpcore::chamber.cmapscale, DIM, DIM);
}

void Wavefunction::MovieFrame()
{
    int DIM = gpcore::chamber.DIM;
    int DS = DIM * DIM;
    movieFrameLauncher(devMovieBuffer, devPsi, gpcore::chamber.cmapscale, DIM, DIM);
    checkCudaErrors(cudaMemcpy(hostMovieBuffer, devMovieBuffer, sizeof(int)*DS, cudaMemcpyDeviceToHost));

    fwrite(hostMovieBuffer, sizeof(int), DIM*DIM, ffmpeg);

}


void Wavefunction::RealSpaceHalfStep(int timestep) {
    realspaceKernelLauncher(
                    devPsi, 
                    gpcore::chamber.devExpPotential, 
                    devPsi, 
                    gpcore::chamber.g[timestep],
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
    cudaDeviceSynchronize();
}

void Wavefunction::Renormalize() {
    parSum(devPsi, devDensity, gpcore::chamber.dx, DIM, DIM);
}

// void Wavefunction::CalculateEnergy(double energy) {
//     energy = 0.;
//     energyKernelLauncher(devPsi, gpcore::chamber.)
// }

void Wavefunction::RotatingFrame(unsigned long timestep, unsigned long steps) {
    int DIM = gpcore::chamber.DIM;
    double omega;
    if (steps==0) {
        omega = gpcore::chamber.omegaR[0];
    } else {
        omega = gpcore::chamber.omegaR[timestep];
    }
    double renorm1D = 1.0 / pow(DIM, 0.5);
    double renorm2D = 1.0 / DIM;
    

    if ((timestep==0) || (omega != gpcore::chamber.omegaR[timestep-1])) {
        gaugefieldKernelLauncher(omega,
                                 gpcore::chamber.devXkY, gpcore::chamber.devYkX,
                                 gpcore::chamber.devExpXkY, gpcore::chamber.devExpYkX,
                                 gpcore::chamber.dt, gpcore::chamber.useReal, gpcore::chamber.cooling,
                                 DIM, DIM);
    }

    switch(timestep%2){

        case 0: // even step
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_FORWARD); // xky
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm
        momentumspaceKernelLauncher(devPsi, gpcore::chamber.devExpXkY, devPsi, DIM, DIM);
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_INVERSE);
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm

        cufftExecZ2Z(gpcore::chamber.fftPlan2D,devPsi,devPsi,CUFFT_FORWARD); //2D forward
        multKernelLauncher(devPsi, renorm2D, DIM, DIM); // fft renorm
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_INVERSE); //1D inverse to ykx
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm;
        momentumspaceKernelLauncher(devPsi, gpcore::chamber.devExpYkX, devPsi, DIM, DIM);
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_FORWARD); // kxky 
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm;
        cufftExecZ2Z(gpcore::chamber.fftPlan2D,devPsi,devPsi,CUFFT_INVERSE); //2D Inverse
        multKernelLauncher(devPsi, renorm2D, DIM, DIM); // fft renorm
        cudaDeviceSynchronize();
        break;

        case 1:	// odd step
        cufftExecZ2Z(gpcore::chamber.fftPlan2D,devPsi,devPsi,CUFFT_FORWARD); //2D forward
        multKernelLauncher(devPsi, renorm2D, DIM, DIM); // fft renorm
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_INVERSE); //1D inverse to ykx
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm;
        momentumspaceKernelLauncher(devPsi, gpcore::chamber.devExpYkX, devPsi, DIM, DIM);
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_FORWARD); // kxky 
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm;
        cufftExecZ2Z(gpcore::chamber.fftPlan2D,devPsi,devPsi,CUFFT_INVERSE); //2D Inverse
        multKernelLauncher(devPsi, renorm2D, DIM, DIM); // fft renorm

        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_FORWARD); // xky
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm;
        momentumspaceKernelLauncher(devPsi, gpcore::chamber.devExpXkY, devPsi, DIM, DIM);
        cufftExecZ2Z(gpcore::chamber.fftPlan1D,devPsi,devPsi,CUFFT_INVERSE);
        multKernelLauncher(devPsi, renorm1D, DIM, DIM); // fft renorm;
        cudaDeviceSynchronize();
        break;
    }

    
}

void Wavefunction::ExportToVariable(cuDoubleComplex *arr)
{
    int DIM = gpcore::chamber.DIM;
    int DS = DIM * DIM;
    checkCudaErrors(cudaMemcpy(hostPsi, devPsi, sizeof(cuDoubleComplex) * DS, cudaMemcpyDeviceToHost));
    memcpy(arr, hostPsi, sizeof(cuDoubleComplex) * DS);
}

void Wavefunction::MovieCleanup()
{
    pclose(ffmpeg);
    free(hostMovieBuffer);
    checkCudaErrors(cudaFree(devMovieBuffer));
}

void Wavefunction::Cleanup()
{
    free(hostPsi);
    checkCudaErrors(cudaFree(devPsi));
    checkCudaErrors(cudaFree(devDensity));
}