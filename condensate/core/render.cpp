
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
#include "defines.h"
#include "gp_kernels.h"
#include "helper_cudagl.h"
#include "render.hpp"
#include "initialize.hpp"


namespace render {


    // Timer stuff
    static int fpsCount = 0;
    static int fpsLimit = 1;
    StopWatchInterface *timer = NULL;

    // texture and pixel objects
    static GLuint pbo = 0; // OpenGL pixel buffer object
    GLuint tex = 0; // OpenGL texture object
    struct cudaGraphicsResource *cuda_pbo_resource;

    // Use a class to store these. then pass the class in?

    void startTimer() {
        sdkCreateTimer(&timer);
        sdkResetTimer(&timer);
    }


    void render() {
        uchar4 *d_out = 0;
        cudaGraphicsMapResources(1, &cuda_pbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void **)&d_out, NULL, cuda_pbo_resource);
        kernelLauncher(d_out, gpcore::devPsi, DIM, DIM);
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
        free(gpcore::hostPsi);
        checkCudaErrors(cudaFree(gpcore::devPsi));
        checkCudaErrors(cudaFree(gpcore::hostV));
        sdkDeleteTimer(&timer);
        checkCudaErrors(cudaDeviceReset());

    }

    int initGL() {

        int argc = 1;
        char *argv[1] = {(char*)"Something"};
        glutInit(&argc, argv);
        glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE);
        glutInitWindowSize(gpcore::wWidth, gpcore::wHeight);
        glutCreateWindow("Compute Stable Fluids");

        glutDisplayFunc(display);
        glutKeyboardFunc(keyboard);
        glutCloseFunc(cleanup);
        glewInit();
        gluOrtho2D(0, DIM, DIM, 0);
        
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

    void startOpenGL(){

        if (glutGetWindow()==0){
            initGL();
        } 
        startTimer();
        initPixelBuffer();
    }

}