/* File: gpcore.i */
%module gpcore

%{
    #define SWIG_FILE_WITH_INIT
    #include <cuda.h>
    #include <cufft.h>
    #include "gpcore.hpp"
%}
// Note: if numpy.i gives errors, make sure there is a symlink from ~/.local/lib/python3.6/site-packages/numpy/core/include/numpy to /usr/include/numpy

%include "numpy.i"
%init %{
import_array();
%}
%numpy_typemaps(cuDoubleComplex, NPY_CDOUBLE, int)

%apply (int DIM1, int DIM2, cuDoubleComplex* INPLACE_ARRAY2) {(int sizex, int sizey, cuDoubleComplex *arr)};

%include gpcore.hpp

