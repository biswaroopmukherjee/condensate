#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cmath>  
#include <algorithm>

#include "defines.h"
#include "chamber.hpp"
#include "helper_cudagl.h"
#include "gp_kernels.h"


// setup for the size of the grid, the kinetic part, and other parameters
void Chamber::setup(int size, double fovinput, double ginput, double deltat, bool useImag, double cool) {
    DIM = size;
    DS = DIM*DIM;
	dt = deltat;
	mass = 3.8e-26; // Na-23 mass
	spoon1.strength = 0;
	useLeapMotion = false;

	stopSim = false;
	useRotatingFrame = false;
	useImaginaryTime = useImag;
	cooling = useImaginaryTime ? 1 : cool;
	useReal = useImaginaryTime ? 0 : 1;

	int doubleDIM = sizeof(double) * DIM;
    X = (double *) malloc(doubleDIM);
	Y = (double *) malloc(doubleDIM);
	kX = (double *) malloc(doubleDIM);
	kY = (double *) malloc(doubleDIM);

	g = ginput;
	mass = 3.8e-26; //Rb 87 mass, kg
	fov = fovinput/2;
	kfov = (PI/fov)*(DIM>>1);
    dx = fov/(DIM>>1);
    dk = PI/(fov);
    
	unsigned int i, j; 
    for(i=0; i<DIM/2; ++i){
		X[i] = -fov + (i+1)*dx;
		X[i + (DIM/2)] = (i+1)*dx;

		Y[i] = -fov + (i+1)*dx;
		Y[i + (DIM/2)] = (i+1)*dx;

		kX[i] = (i+1)*dk;
		kX[i + (DIM/2)] = -kfov + (i+1)*dk;

		kY[i] = (i+1)*dk;
		kY[i + (DIM/2)] = -kfov + (i+1)*dk;
	}

    int doubleDS = sizeof(double) * DS;
	int cudoubleDS = sizeof(cuDoubleComplex) * DS;

	Potential = (double *) malloc(doubleDS);
	Kinetic   = (double *) malloc(doubleDS);
    XkY = (double *) malloc(doubleDS);
	YkX = (double *) malloc(doubleDS);
	hostExpKinetic   = (cuDoubleComplex *) malloc(cudoubleDS);
	hostExpPotential = (cuDoubleComplex *) malloc(cudoubleDS);
    
	cudaMalloc((void**) &devPotential, doubleDS);
	cudaMalloc((void**) &devExpPotential, cudoubleDS);
	cudaMalloc((void**) &devExpKinetic, cudoubleDS);
	cudaMalloc((void**) &devXkY, doubleDS);
	cudaMalloc((void**) &devYkX, doubleDS);
	cudaMalloc((void**) &devExpXkY, cudoubleDS);
	cudaMalloc((void**) &devExpYkX, cudoubleDS);
	
	// for ffts;
    cufftPlan2d(&fftPlan2D, DIM, DIM, CUFFT_Z2Z);
    cufftPlan1d(&fftPlan1D, DIM, CUFFT_Z2Z, DIM);


	for( i=0; i < DIM; i++ ){
		for( j=0; j < DIM; j++ ){

			Kinetic[(i*DIM + j)] = (HBAR*HBAR/(2*mass)) * (kX[i]*kX[i] + kY[j]*kY[j]);
			hostExpKinetic[(i*DIM + j)].x = exp( -Kinetic[(i*DIM + j)] * (cooling*dt/HBAR) ) *
										    cos( -Kinetic[(i*DIM + j)] * (useReal*dt/HBAR) );
			hostExpKinetic[(i*DIM + j)].y = exp( -Kinetic[(i*DIM + j)] * (cooling*dt/HBAR) ) *
										    sin( -Kinetic[(i*DIM + j)] * (useReal*dt/HBAR) );
										  
			XkY[(i*DIM + j)] =  X[i]*kY[j];
			YkX[(i*DIM + j)] = -Y[j]*kY[i];

		}
	}

	// Copy to device
    checkCudaErrors(cudaMemcpy(devExpKinetic, hostExpKinetic, cudoubleDS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devXkY, XkY, doubleDS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devYkX, YkX, doubleDS, cudaMemcpyHostToDevice));


}

// setup for harmonic potential: todo: move to separate objects
void Chamber::setHarmonicPotential(double o, double ep) {
    omega = o;
	epsilon = ep;
	unsigned int i, j;
	for( i=0; i<DIM; i++ ){
		for( j=0; j<DIM; j++){
			Potential[(i*DIM + j)] = 0.5 * mass * ( (1-epsilon) * pow(omega * X[(i)], 2) +
												    (1+epsilon) * pow(omega * Y[(j)], 2) ) ;
			hostExpPotential[(i*DIM + j)].x = exp( -Potential[(i*DIM + j)] * cooling*dt/(2*HBAR)) *
											  cos( -Potential[(i*DIM + j)] * useReal*dt/(2*HBAR));
			hostExpPotential[(i*DIM + j)].y = exp( -Potential[(i*DIM + j)] * cooling*dt/(2*HBAR)) *
											  sin( -Potential[(i*DIM + j)] * useReal*dt/(2*HBAR));
		}
	}
	// Copy to device
	checkCudaErrors(cudaMemcpy(devPotential, Potential, sizeof(double) * DS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devExpPotential, hostExpPotential, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));
};

void Chamber::AbsorbingBoundaryConditions(double strength, double radius) {
	unsigned int i, j;
	double V_abc, rfromcenter, box_sharpness;
	box_sharpness = radius/10;
	for( i=0; i<DIM; i++ ){
		for( j=0; j<DIM; j++){
			rfromcenter = pow((pow(X[i],2) + pow(X[j],2)),0.5);
			V_abc = strength * 1e-5 * 0.5*mass*( erf( (rfromcenter-radius)/box_sharpness) +1 );
			hostExpPotential[(i*DIM + j)].x *= exp( -V_abc * useReal*dt/(2*HBAR));
			hostExpPotential[(i*DIM + j)].y *= exp( -V_abc * useReal*dt/(2*HBAR));
		}
	}
	// Copy to device
    checkCudaErrors(cudaMemcpy(devExpPotential, hostExpPotential, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));
}

void Chamber::SetupSpoon(double strength, double radius) {
	spoon1.strength = 1e-6 * 0.5 * mass * strength;
	spoon1.strengthSetting = spoon1.strength; // this is changed depending on if the spoon is on or not
	spoon1.radius = (int) floor(radius / dx);
	spoon1.pos.x = 0;
	spoon1.pos.y = 0;
}

void Chamber::Spoon() {
    spoonKernelLauncher(devPotential, devExpPotential, spoon1, dt, useReal, cooling, DIM, DIM); // kernel to rewrite spoon and devexppotential
}

void Chamber::Cleanup()
{
	cudaDeviceSynchronize();
    free(Kinetic); free(hostExpKinetic); 
	free(Potential); free(hostExpPotential);
	if (useRotatingFrame) free(omegaR);
	free(XkY); free(YkX);
	free(X); free(Y);
	free(kX); free(kY);
    cudaFree(devExpPotential);
    cudaFree(devExpKinetic);
	cudaFree(devPotential);
    cudaFree(devXkY);
    cudaFree(devYkX);
    cudaFree(devExpXkY);
    cudaFree(devExpYkX);
	cufftDestroy(fftPlan1D); cufftDestroy(fftPlan2D);
    checkCudaErrors(cudaDeviceReset());
}