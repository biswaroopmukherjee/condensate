
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>  
#include <chrono>

// CUDA+OPENGL includes
#include <GL/glew.h>
#include <algorithm>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <GL/freeglut.h>

#include "gpcore.hpp"
#include "render.hpp"
#include "wavefunction.hpp"
#include "chamber.hpp"
#include "defines.h"
#include "gp_kernels.h"
#include "helper_cudagl.h"

namespace gpcore {
  Chamber chamber;
  Wavefunction Psi;
}


// Set the spatial grid size of the system
void setup(int size, double deltat, double time, double omega_r, bool useImag, double cool) {
  gpcore::chamber.setup(size, deltat, time, omega_r, useImag, cool);
}


// Set the parameters of the harmonic potential, and use that as the current potential
void setHarmonicPotential(double omega, double epsilon) {
  gpcore::chamber.setHarmonicPotential(omega, epsilon);
  printf("\n\n x = %e\n\n", gpcore::chamber.X[(2*512 + 120)]);
  printf("\n\n pot = %e\n\n", gpcore::chamber.Potential[(2*512+120)]);
}


void getPotential(int sizeX, int sizeY, double *V){
  unsigned int i;
  for( i=0; i<gpcore::chamber.DS; i++ ) {
    V[i] = gpcore::chamber.Potential[i] / HBAR;
  }
}

// evolve the wavefunction
void Evolve(int sizex, int sizey, cuDoubleComplex *arr) {

  int speed = 6000;
  int print = 5;
  printf("\n\n Starting GP... \n\n");

  render::startOpenGL();

  gpcore::Psi.Initialize(arr);

  printf("\n\n speed = %i\n\n", speed);
  printf("\n\n omega_r = %.3f\n\n", gpcore::chamber.omegaRotation);
  printf("\n\n pot = %.3f\n\n", gpcore::chamber.Potential[250*512 + 250]);


  for( int a = 0; a < gpcore::chamber.timesteps; a++ ) {

      double mult = exp(-(double)a / speed);
      // gpcore::Psi.RealSpaceHalfStep(); 
      // gpcore::Psi.MomentumSpaceStep();
      // gpcore::Psi.RealSpaceHalfStep();
      // gpcore::Psi.RotatingFrame();
      gpcore::Psi.Step(mult);
      if (a%print ==0){

        printf("\n\n mult = %.3f", mult);
        
        
        glutMainLoopEvent();
      }
   }



  gpcore::Psi.ExportToVariable(arr);
  gpcore::Psi.Cleanup();
  gpcore::chamber.Cleanup();
  render::cleanup();

  printf("\n Done \n\n");
}