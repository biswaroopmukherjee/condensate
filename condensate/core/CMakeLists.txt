# GP-realtime: a realtime GP simulator
# 2019 Biswaroop Mukherjee

cmake_minimum_required(VERSION 3.10)
project (gpcore)

# packages
find_package(CUDA REQUIRED)
find_package(SWIG REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLUT REQUIRED)

# SWIG stuff
INCLUDE(${SWIG_USE_FILE})

FIND_PACKAGE(PythonLibs)
INCLUDE_DIRECTORIES(${PYTHON_INCLUDE_PATH})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC"  )
set(CMAKE_POSITION_INDEPENDENT_CODE ON)


# libraries
cuda_add_library(gpcoregpu
    gpcore.hpp
    gpcore.cpp

    initialize.hpp
    initialize.cpp

    render.hpp
    render.cpp

    helper_cudagl.h
    gp_kernels.h
    gp_kernels.cu

)
cuda_add_cufft_to_target(gpcoregpu)


target_link_libraries(gpcoregpu 
    ${OPENGL_LIBRARIES} ${GLEW_LIBRARIES} ${GLUT_LIBRARIES})

target_include_directories(gpcoregpu PUBLIC
    ${OPENGL_INCLUDE_DIR}
    ${GLUT_INCLUDE_DIR}
)

SET_PROPERTY(SOURCE gpcore.i PROPERTY CPLUSPLUS ON)
SWIG_ADD_LIBRARY(gpcore LANGUAGE python SOURCES gpcore.i)
SWIG_LINK_LIBRARIES(gpcore gpcoregpu ${PYTHON_LIBRARIES})







