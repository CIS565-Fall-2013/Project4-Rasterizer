// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void initLights();
void initBuffers(glm::vec2 resolution);
void clearBuffers(glm::vec2 resolution);
void drawToStencilBuffer(glm::vec2 resolution, glm::vec3 eye, glm::vec3 center, float* vbo, int vbosize, int* ibo, int ibosize, int stencil);
void clearOnStencil(glm::vec2 resolution, int stencil);
void cudaRasterizeCore(glm::vec2 resolution, glm::vec3 eye, glm::vec3 center,
											 float* vbo, int vbosize, float* cbo, int cbosize, float* nbo,
											 int nbosize, int* ibo, int ibosize, bool stencilTest, bool perPrimitive, int stencil);
void renderToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3 eye);
void kernelCleanup();
void freeBuffers();

#endif //RASTERIZEKERNEL_H
