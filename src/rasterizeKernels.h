// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif


#define BACKCULLING 1
#define ANTIALIASING 1

void kernelCleanup();
void cudaRasterizeCore(glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, 
					   glm::mat4 modelViewProjection, glm::mat4 viewPort, glm::vec4 lightPos, glm::vec3 cameraPosition, glm::vec3 lookAt, bool isStencil, int first, int second, char keyValue, int* stencilBuffer);

void initalKernel(glm::vec2 resolution, int* stencilBuffer);
void renderKernel(uchar4* PBOpos, glm::vec2 resolution);



#endif //RASTERIZEKERNEL_H
