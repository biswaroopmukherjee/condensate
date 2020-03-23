
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
#include "leap_controls.hpp"

namespace gpcore {
  Chamber chamber;
  Wavefunction Psi;
}


// Set the spatial grid size of the system
void Setup(int size, double fov, double g, double deltat, bool useImag, double cool) {
  gpcore::chamber.setup(size, fov, g, deltat, useImag, cool);
}


// Set the parameters of the harmonic potential, and use that as the current potential
void SetHarmonicPotential(double omega, double epsilon) {
  gpcore::chamber.setHarmonicPotential(omega, epsilon);
}


// Extract the potential. Todo: import potential, timedependent: make an object, like wavefunction
void GetPotential(int sizeX, int sizeY, double *V){
  for( unsigned int i=0; i<gpcore::chamber.DS; i++ ) {
    V[i] = gpcore::chamber.Potential[i] / HBAR;
  }
}

// Setup rotating frame with varying rotation freq omegaR. 
void RotatingFrame(int size, double *omega_r){
  gpcore::chamber.useRotatingFrame = true;
  gpcore::chamber.omegaR = (double *) malloc(sizeof(double) * size);
  for( unsigned int i=0; i<size; i++ ) {
    gpcore::chamber.omegaR[i] = omega_r[i];
  }
};

// Set up absorbing boundary conditions at a radius
void AbsorbingBoundaryConditions(double strength, double radius) {
  gpcore::chamber.AbsorbingBoundaryConditions(strength, radius);
}

// Set up a spoon
void SetupSpoon(double strength, double radius) {
  gpcore::chamber.SetupSpoon(strength, radius);
}

// Set up a leap motion tracker
void SetupLeapMotion(double centerx, double centery, double zoomx, double zoomy) {
  gpcore::chamber.useLeapMotion = true;
  gpcore::chamber.LeapProps = {centerx,centery, zoomx, zoomy};
  printf("\ncenter = %.3f\n", gpcore::chamber.LeapProps.x);
}

// evolve the wavefunction
void Evolve(int sizex, int sizey, cuDoubleComplex *arr, unsigned long steps, int skip, bool show, double vmax) {

  printf("\n\n Starting GP... \n\n");
  if (show) render::startOpenGL();
  gpcore::Psi.Initialize(arr);
  gpcore::chamber.cmapscale = vmax;
  unsigned long a=0;

  Controller controller;


  while (!gpcore::chamber.stopSim) 
  {
    // printf("\n x=%i\n", gpcore::chamber.spoon1.pos.x);
    gpcore::Psi.RealSpaceHalfStep(); 
    gpcore::Psi.MomentumSpaceStep();
    gpcore::Psi.RealSpaceHalfStep();
    if (gpcore::chamber.useRotatingFrame) gpcore::Psi.RotatingFrame(a, steps);
    if (gpcore::chamber.useImaginaryTime || (gpcore::chamber.cooling!=0)) gpcore::Psi.Renormalize();
    if (gpcore::chamber.spoon1.strength != 0) gpcore::chamber.Spoon();
    if ((a%skip == 0) && show) glutMainLoopEvent();
    a++;
    if (a==steps) gpcore::chamber.stopSim = true; 
    
    if (gpcore::chamber.useLeapMotion) printfingers(controller);



   }
  
  if (show) render::cleanup();
  gpcore::Psi.ExportToVariable(arr);
  gpcore::Psi.Cleanup();
  gpcore::chamber.Cleanup();

  printf("\n Done \n\n");
}
