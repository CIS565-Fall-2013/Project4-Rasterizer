// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "utilities.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define checkCUDAErrorWithLine(msg)

//-------------------------------
//---------CAMERA STUFF----------
//-------------------------------
//stores cam data needed for viewport transformation
struct camera{
	glm::vec3 eye;
	glm::vec3 up;
	glm::vec3 center;
	float fov;
	float zNear;
	float zFar;
};

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, 
					   camera& cam);

#endif //RASTERIZEKERNEL_H
