// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include <time.h>

#include <windows.h>
#include "glm/glm.hpp"
#include "rasterizeStructs.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define MAX_DEPTH 10000.0f
#define DEPTH_EPSILON 0.0000000000001f


void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* nbo, int nbosize, 
					   float* cbo, int cbosize, int* ibo, int ibosize, uniforms viewMats, pipelineOpts opts, PerformanceMetrics &metrics);

#endif //RASTERIZEKERNEL_H
