// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"

enum shadeMode {COLOR,LIGHTING,SLIGHTING,NORMALS,ZDEPTH} ;


#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,float* nbo, int nbosize,const glm::mat4& modelMatrix, const glm::mat4& viewMatrix, const glm::mat4& projectionMatrix, const glm::vec2 zplanes,shadeMode sm, const glm::vec3 camPos);

#endif //RASTERIZEKERNEL_H
