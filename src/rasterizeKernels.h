// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <cuda.h>
#include <cmath>
#include "rasterizeStructs.h"
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define checkCUDAErrorWithLine(msg) checkCUDAError(msg, __LINE__)
#define BACK_FACE_CULLING 1
#define TURN_TABLE 1
#define ANTIALIASING 0
#define ENABLE_FRAG_SHADER 1

using glm::mat4;
using glm::vec4;
using glm::vec3;
using glm::vec2;
using glm::clamp;
using glm::inverse;
using glm::transpose;
using glm::dot;
using glm::normalize;

void printVAO(float* device_vao, int size);
void kernelCleanup();
void cudaRasterizeCore(camera* cam, uchar4* pos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, light* lights, int lightsize);
void allocateDeviceMemory( float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize, light* lights, int lightsize);
#endif //RASTERIZEKERNEL_H
