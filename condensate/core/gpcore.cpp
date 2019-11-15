
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>  

// CUDA+OPENGL includes
#include <GL/glew.h>
#include <algorithm>
#include <cuda_gl_interop.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <GL/freeglut.h>

#include <chrono>


#include "gpcore.hpp"
#include "defines.h"
#include "gp_kernels.h"
#include "helper_timer.h"


#define checkCudaErrors(ans) { _checkCudaErrors((ans), __FILE__, __LINE__); }
inline void _checkCudaErrors(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      printf("Error: %s:%d, ", file, line);
      printf("code:%d, reason: %s\n", code, cudaGetErrorString(code));
      if (abort) exit(code);
   }
}




// Physics and FFT plans
cufftHandle planr2c;
cufftHandle planc2r;

static float2 *hostV = NULL;
float2 *hostPsi = NULL;
float2 *devPsi = NULL;
static int wWidth = std::max(512, DIM);
static int wHeight = std::max(512, DIM);

// Timer stuff
static int fpsCount = 0;
static int fpsLimit = 1;
StopWatchInterface *timer = NULL;

// texture and pixel objects
static GLuint pbo = 0; // OpenGL pixel buffer object
GLuint tex = 0; // OpenGL texture object
struct cudaGraphicsResource *cuda_pbo_resource;


void render() {
  uchar4 *d_out = 0;
  cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
  cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
  kernelLauncher(d_out, devPsi, DIM, DIM);
  cudaGraphicsUnmapResources(1, &cuda_pbo_resource, 0);
}

void drawTexture() {
  glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, DIM, DIM, 0, GL_RGBA, GL_UNSIGNED_BYTE, NULL);
  glEnable(GL_TEXTURE_2D);
  glBegin(GL_QUADS);
  glTexCoord2f(0.0f, 0.0f); glVertex2f(0, 0);
  glTexCoord2f(0.0f, 1.0f); glVertex2f(0, DIM);
  glTexCoord2f(1.0f, 1.0f); glVertex2f(DIM, DIM);
  glTexCoord2f(1.0f, 0.0f); glVertex2f(DIM, 0);
  glEnd();
  glDisable(GL_TEXTURE_2D);
}

void display(void) {

  sdkStartTimer(&timer);
  render();
  drawTexture();
  sdkStopTimer(&timer);// Finish timing before swap buffers to avoid refresh sync
  glutSwapBuffers();

  fpsCount++;

  if (fpsCount == fpsLimit) {
    char fps[256];
    float ifps = 1.f / (sdkGetAverageTimerValue(&timer) / 1000.f);
    sprintf(fps, "Cuda/GL GP Wavefunction (%d x %d): %3.1f fps", DIM, DIM, ifps);
    glutSetWindowTitle(fps);
    fpsCount = 0;
    fpsLimit = (int)std::max(ifps, 1.f);
    sdkResetTimer(&timer);
  }

  glutPostRedisplay();
}


void initPsi(float2 *p, int dx, int dy) {
  int i, j;

  for (i = 0; i < dy; i++) {
    for (j = 0; j < dx; j++) {
      p[i * dx + j].x = exp(-( pow( (i-200.)/200., 2. ) + pow( (j-200.)/200., 2. ) ) );
      p[i * dx + j].y = exp(-( pow( (i-200.)/200., 2. ) + pow( (j-200.)/200., 2. ) ) );
    }
  }
}

void keyboard(unsigned char key, int x, int y) {
  switch (key) {
  case 27:
    glutDestroyWindow(glutGetWindow());
    return;
    break;

  default:
    break;
  }
}


void cleanup(void) 
{
  if (pbo) {
      cudaGraphicsUnregisterResource(cuda_pbo_resource);
      glDeleteBuffers(1, &pbo);
      glDeleteTextures(1, &tex);
  }
  // Free all host and device resources
  free(hostPsi);
  cudaFree(devPsi);
  cudaFree(hostV);
  cufftDestroy(planr2c);
  cufftDestroy(planc2r);
  sdkDeleteTimer(&timer);
  cudaDeviceReset();

}

int initGL() {

  int argc = 1;
  char *argv[1] = {(char*)"Something"};
  glutInit(&argc, argv);
  glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
  glutInitWindowSize(wWidth, wHeight);
  glutCreateWindow("Compute Stable Fluids");

  glutDisplayFunc(display);
  glutKeyboardFunc(keyboard);
  glutCloseFunc(cleanup);
  glewInit();

  if (!glewIsSupported("GL_ARB_pixel_buffer_object")) {
    fprintf(stderr, "ERROR: Support for necessary OpenGL extensions missing.");
    fflush(stderr);
    return false;
  }

  return true;
}

void initPixelBuffer() {
    glGenBuffers(1, &pbo);
    glBindBuffer(GL_PIXEL_UNPACK_BUFFER, pbo);
    glBufferData(GL_PIXEL_UNPACK_BUFFER, 4*DIM*DIM*sizeof(GLubyte), 0, GL_DYNAMIC_DRAW);
    glGenTextures(1, &tex);
    glBindTexture(GL_TEXTURE_2D, tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    cudaGraphicsGLRegisterBuffer(&cuda_pbo_resource, pbo, cudaGraphicsMapFlagsWriteDiscard);
}

double fft2d(int speed, int print) {

  printf("\n\n Starting GP... \n\n");

  if (glutGetWindow()==0){
    initGL();
    gluOrtho2D(0, DIM, DIM, 0);
  } else {
  }


  sdkCreateTimer(&timer);
  sdkResetTimer(&timer);

  // Create Wavefunction
  hostPsi = (float2 *)malloc(sizeof(float2) * DS);
  memset(hostPsi, 0, sizeof(float2) * DS);
  initPsi(hostPsi, DIM, DIM);

  // Allocate and initialize device data
  checkCudaErrors(cudaMalloc((void **)&devPsi, sizeof(float2) * DIM * DIM));
  checkCudaErrors(cudaMemcpy(devPsi, hostPsi, sizeof(float2) * DS, cudaMemcpyHostToDevice));

  initPixelBuffer();

  auto start = std::chrono::high_resolution_clock::now();

  for( int a = 0; a < 200; a++ ) {
      float mult = exp(-(float)a / speed);
      // printf("%.3f\n", mult);
      multKernel(devPsi, mult, DIM, DIM); 
      if (a % print == 0){
        glutMainLoopEvent();
      }
   }
  auto finish = std::chrono::high_resolution_clock::now(); 
  std::chrono::duration<double> elapsed = finish - start;
  double out=0;
  out += elapsed.count() / 1e-3;

  printf("%i\n\n", glutGetWindow());

  checkCudaErrors(cudaDeviceReset());
  // glutDestroyWindow(glutGetWindow());
  // glutLeaveMainLoop();

  printf("\n Done \n\n");
  printf("%i\n\n", glutGetWindow());

  return out;
}

void getwindow(void)
{
  printf("%i\n\n", glutGetWindow());
}
