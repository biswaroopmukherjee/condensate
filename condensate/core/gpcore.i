/* File: gpcore.i */
%module gpcore

%{
#define SWIG_FILE_WITH_INIT
#include "gpcore.hpp"
%}

double fft2d(int speed, int print);
void getwindow(void);