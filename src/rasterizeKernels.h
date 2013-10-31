// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#ifndef RASTERIZEKERNEL_H
#define RASTERIZEKERNEL_H

#include <stdio.h>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <cuda.h>
#include <cmath>
#include "glm/glm.hpp"
#include "glm/gtc/matrix_transform.hpp"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

#define DEBUG 0
#define DEPTHPRECHECK 0
#define OPTIMIZE_RASTER 1
// Will only work on copmute 3.5+, so check with someone
#define DYNAMICPARALLELISM 0
#define PERFANALYZE 1
#define BACKFACECULLING 1
#define BACKFACEIGNORING 1

#define LIGHTPOS glm::vec3(5000,5000,20000)

#define vertexStride 4
#define colorStride 3
#define indexStride 3

class cam
{
	public:
	float rad;
	float theta, phi;
	bool idle;
	float delPhi;
	glm::vec3 pos;
	cam();
	void reset();
	void setFrame();
};

extern cam mouseCam;

void kernelCleanup();
void cudaRasterizeCore(uchar4* pos, glm::vec2 resolution, float frame, float* vbo, float* nbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize);

#endif //RASTERIZEKERNEL_H
