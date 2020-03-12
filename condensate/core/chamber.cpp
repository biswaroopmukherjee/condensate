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


// setup for the size of the grid, the kinetic part, and other parameters
void Chamber::setup(int size, double deltat, double time, double omega_r, bool useImag, double cool) {
    DIM = size;
    DS = DIM*DIM;
	timesteps = time;
	dt = deltat;

	useImaginaryTime = useImag;
	cooling = useImaginaryTime ? 1 : cool;
	useReal = useImaginaryTime ? 0 : 1;

	int doubleDIM = sizeof(double) * DIM;
    X = (double *) malloc(doubleDIM);
	Y = (double *) malloc(doubleDIM);
	kX = (double *) malloc(doubleDIM);
	kY = (double *) malloc(doubleDIM);

    atomNumber = 1e5;
	mass = 1.4431607e-25; //Rb 87 mass, kg
	a_s = 4.67e-9;
    omegaZ = 10;
    omega = 1;
	omegaRotation = omega_r;
	Rxy = pow(15,0.2)*pow(atomNumber*a_s*sqrt(mass*omegaZ/HBAR),0.2);
	a0 = sqrt(HBAR/(2*mass*omega));
	fov = 7*Rxy*a0;
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
    hostExpXkY = (cuDoubleComplex *) malloc(cudoubleDS);
	hostExpYkX = (cuDoubleComplex *) malloc(cudoubleDS);
	hostExpKinetic   = (cuDoubleComplex *) malloc(cudoubleDS);
	hostExpPotential = (cuDoubleComplex *) malloc(cudoubleDS);

    
	cudaMalloc((void**) &devExpPotential, cudoubleDS);
	cudaMalloc((void**) &devExpKinetic, cudoubleDS);
	cudaMalloc((void**) &devExpXkY, cudoubleDS);
	cudaMalloc((void**) &devExpYkX, cudoubleDS);
    cufftPlan2d(&fftPlan2D, DIM, DIM, CUFFT_Z2Z);
	// cudaMalloc((void**) &par_sum, sizeof(cuDoubleComplex) * (DS/threads));
	
	// for ffts
	double renorm_factor_2d=1.0/pow(DS,0.5);
	double renorm_factor_1d=1.0/pow(DIM,0.5);


	for( i=0; i < DIM; i++ ){
		for( j=0; j < DIM; j++ ){

			Kinetic[(i*DIM + j)] = (HBAR*HBAR/(2*mass)) * (kX[i]*kX[i] + kY[j]*kY[j]);
			hostExpKinetic[(i*DIM + j)].x = exp( -Kinetic[(i*DIM + j)] * (cooling*dt/HBAR) ) *
										    cos( -Kinetic[(i*DIM + j)] * (useReal*dt/HBAR) );
			hostExpKinetic[(i*DIM + j)].y = exp( -Kinetic[(i*DIM + j)] * (cooling*dt/HBAR) ) *
										    sin( -Kinetic[(i*DIM + j)] * (useReal*dt/HBAR) );
										  
			XkY[(i*DIM + j)] = X[i]*kY[j];
			YkX[(i*DIM + j)] = -Y[j]*kY[i];
			hostExpXkY[(i*DIM + j)].x = cos(-omegaRotation*XkY[(i*DIM + j)]*dt);
			hostExpXkY[(i*DIM + j)].y = sin(-omegaRotation*XkY[(i*DIM + j)]*dt);
			hostExpYkX[(i*DIM + j)].x = cos(-omegaRotation*YkX[(i*DIM + j)]*dt);
			hostExpYkX[(i*DIM + j)].y = sin(-omegaRotation*YkX[(i*DIM + j)]*dt);

		}
	}

	// Copy to device
    checkCudaErrors(cudaMemcpy(devExpKinetic, hostExpKinetic, cudoubleDS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devExpXkY, hostExpXkY, cudoubleDS, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(devExpYkX, hostExpYkX, cudoubleDS, cudaMemcpyHostToDevice));


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
    checkCudaErrors(cudaMemcpy(devExpPotential, hostExpPotential, sizeof(cuDoubleComplex) * DS, cudaMemcpyHostToDevice));
};





void Chamber::Cleanup()
{
    free(Kinetic); free(hostExpKinetic); 
	free(hostExpXkY); free(hostExpYkX); 
	free(Potential); free(hostExpPotential);
    // checkCudaErrors(cudaFree(devExpPotential));
    // checkCudaErrors(cudaFree(devExpKinetic));
    // checkCudaErrors(cudaFree(devExpXkY));
    // checkCudaErrors(cudaFree(devExpYkX));
    checkCudaErrors(cudaDeviceReset());
}