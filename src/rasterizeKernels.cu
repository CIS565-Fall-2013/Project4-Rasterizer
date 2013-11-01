// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "glm/gtc/matrix_transform.hpp"
#define NATHANS_EPSILON 0.0001

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
//float* tmp_zbuffer;
int* framebuffer_writes; //keeps track of how many writes to the framebuffer we've made.
float* device_vbo;
float* device_nbo;
float* modelspace_vbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
    exit(EXIT_FAILURE); 
  }
} 

//Handy dandy little hashing function that provides seeds for random number generation
__host__ __device__ unsigned int hash(unsigned int a){
    a = (a+0x7ed55d16) + (a<<12);
    a = (a^0xc761c23c) ^ (a>>19);
    a = (a+0x165667b1) + (a<<5);
    a = (a+0xd3a2646c) ^ (a<<9);
    a = (a+0xfd7046c5) + (a<<3);
    a = (a^0xb55a4f09) ^ (a>>16);
    return a;
}

//atomic min for floats. Got the idea from NVIDIA forums: https://devtalk.nvidia.com/default/topic/492068/atomicmin-with-float/
//but I reimplemented it myself by taking the code for atomicAdd in the class slides and modifying it myself (without looking at what was
//in the forum post).	
//It probably came out almost identical to the one in the forum post anyway, since the forum post and the class slides
//are based on the same example
//__device__ float fatomicMax(float *address, float val)
//{
//	float old = *address, assumed;
//	do {
//		if( old > val ){ //value is not greater (closer to eye) than the old one.
//			return old;
//		}
//		assumed = old;
//		old = atomicCAS(address, assumed, val);
//	} while (assumed != old);
//	return old;
//}

__device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
	int index = (y*resolution.x) + x;
	//if(x >= 0 && y >= 0 && x<resolution.x && y<resolution.y){
	//	//fatomicMax(&tmp_depthbuffer[index], frag.position.z);
	//	//atomicMax(&tmp_depthbuffer[index], frag.position.z);
	//}
	//__threadfence();
	if(x >= 0 && y >= 0 && x<resolution.x && y<resolution.y){
		//if(frag.position.z == tmp_depthbuffer[index]){//if we are indeed the fragment with min Z, then write
		//	int leet = 1337;
		//}
		//printf("Frag z: %f\n", frag.position.z);
		if(depthbuffer[index].position.z < frag.position.z){
			depthbuffer[index] = frag;
		}
	}
}

//Writes a given fragment to a fragment buffer at a given location
__device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution, int* writeCount, bool checkWriteCount){
	int index = (y*resolution.x) + x;
	//if(x >= 0 && y >= 0 && x<resolution.x && y<resolution.y){
	//	//fatomicMax(&tmp_depthbuffer[index], frag.position.z);
	//	//atomicMax(&tmp_depthbuffer[index], frag.position.z);
	//}
	//__threadfence();
	if(x >= 0 && y >= 0 && x<resolution.x && y<resolution.y){
		//if(frag.position.z == tmp_depthbuffer[index]){//if we are indeed the fragment with min Z, then write
		//	int leet = 1337;
		//}
		//printf("Frag z: %f\n", frag.position.z);
		//if(checkWriteCount){
		atomicAdd( &writeCount[index], 1 );
		//}
		if(depthbuffer[index].position.z < frag.position.z){
			depthbuffer[index] = frag;
		}
	}
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ fragment getFromDepthbuffer(int x, int y, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return depthbuffer[index];
  }else{
    fragment f;
    return f;
  }
}

//Writes a given pixel to a pixel buffer at a given location
__host__ __device__ void writeToFramebuffer(int x, int y, glm::vec3 value, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    framebuffer[index] = value;
  }
}

//Reads a pixel from a pixel buffer at a given location
__host__ __device__ glm::vec3 getFromFramebuffer(int x, int y, glm::vec3* framebuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return framebuffer[index];
  }else{
    return glm::vec3(0,0,0);
  }
}

//Kernel that clears a given pixel buffer with a given color
__global__ void clearImage(glm::vec2 resolution, glm::vec3* image, glm::vec3 color){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      image[index] = color;
    }
}

//Kernel that clears a given fragment buffer with a given fragment
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
	  f.triIdx = -1; //no triangle associated.
      buffer[index] = f;
    }
}

//Kernel that writes the image to the OpenGL PBO directly. 
__global__ void sendImageToPBO(uchar4* PBOpos, glm::vec2 resolution, glm::vec3* image){
  
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){

      glm::vec3 color;      
      color.x = image[index].x*255.0;
      color.y = image[index].y*255.0;
      color.z = image[index].z*255.0;

      if(color.x>255){
        color.x = 255;
      }

      if(color.y>255){
        color.y = 255;
      }

      if(color.z>255){
        color.z = 255;
      }
      
      // Each thread writes one pixel location in the texture (textel)
      PBOpos[index].w = 0;
      PBOpos[index].x = color.x;     
      PBOpos[index].y = color.y;
      PBOpos[index].z = color.z;
  }
}

//"xyCoords" are the FLOATING-POINT, sub-pixel-accurate location to be write to
__device__ void writeColorPoint(triangle currTri, int triIdx, glm::vec2 xyCoords, fragment* depthBuffer, glm::vec2 resolution, glm::vec3 color){
	fragment currFrag;
	currFrag.triIdx = triIdx;
	currFrag.color = color; //assume the tri is all one color for now.
	glm::vec3 currBaryCoords = calculateBarycentricCoordinate(currTri, xyCoords);
	float fragZ = getZAtCoordinate(currBaryCoords, currTri) + 0.001f;
	currFrag.position = glm::vec3(xyCoords.x, xyCoords.y, fragZ);
	currFrag.modelPosition = interpVec3(currBaryCoords, currTri.modelspace_p0, currTri.modelspace_p1, currTri.modelspace_p2);
	currFrag.modelNormal = interpVec3(currBaryCoords, currTri.modelspace_n0, currTri.modelspace_n1, currTri.modelspace_n2);
	int pixX = roundf(xyCoords.x);
	int pixY = roundf(xyCoords.y);
	//TODO: incorporate the normal in here **somewhere**
	writeToDepthbuffer((resolution.x - 1) - pixX, (resolution.y - 1) - pixY, currFrag, depthBuffer, resolution);
}


//"xyCoords" are the FLOATING-POINT, sub-pixel-accurate location to be write to
__device__ void writePointInTriangle(triangle currTri, int triIdx, glm::vec2 xyCoords, fragment* depthBuffer, glm::vec2 resolution, bool interpColors, int* writeCount, bool checkWriteCount){
	fragment currFrag;
	currFrag.triIdx = triIdx;
	//currFrag.color = currTri.c0; //assume the tri is all one color for now.
	glm::vec3 currBaryCoords = calculateBarycentricCoordinate(currTri, xyCoords);
	float fragZ = getZAtCoordinate(currBaryCoords, currTri);
	currFrag.position = glm::vec3(xyCoords.x, xyCoords.y, fragZ);
	currFrag.modelPosition = interpVec3(currBaryCoords, currTri.modelspace_p0, currTri.modelspace_p1, currTri.modelspace_p2);
	currFrag.modelNormal = interpVec3(currBaryCoords, currTri.modelspace_n0, currTri.modelspace_n1, currTri.modelspace_n2);
	if(interpColors){
		currFrag.color = interpVec3(currBaryCoords, currTri.c0, currTri.c1, currTri.c2);
	} else { //average the colors. Each face will have a uniform color.
		currFrag.color = (1.0f/3.0f)*(currTri.c0 + currTri.c1 + currTri.c2);
	}
	int pixX = roundf(xyCoords.x);
	int pixY = roundf(xyCoords.y);
	//TODO: incorporate the normal in here **somewhere**
	writeToDepthbuffer((resolution.x - 1) - pixX, (resolution.y - 1) - pixY, currFrag, depthBuffer, resolution, writeCount, checkWriteCount);
}

//rasterize between startX and endX, inclusive
//__device__ int rasterizeHorizLine(glm::vec2 start, glm::vec2 end, fragment* depthBuffer, glm::vec2 resolution, triangle currTri, int triIdx, bool interpColors){
//	int Xinc = roundf(end.x) - roundf(start.x);
//	int sgnXinc = Xinc > 0 ? 1 : -1;
//	int numPixels = abs(Xinc) + 1; //+1 to be inclusive
//	int currX = roundf(start.x);
//	int Y = roundf(start.y); //Y should be the same for the whole line
//	int endY = roundf(end.y);
//	for(int i = 0; i < numPixels; i++){
//		writePointInTriangle(currTri, triIdx, glm::vec2(currX, Y), depthBuffer, resolution, interpColors);
//		if( endY != Y ){
//			writePointInTriangle(currTri, triIdx, glm::vec2(currX, endY), depthBuffer, resolution, interpColors);
//		}
//		currX += sgnXinc; //either increase or decrease currX depending on direction
//	}
//	
//}

//Based on slide 75-76 of the CIS560 notes, Norman I. Badler, University of Pennsylvania. 
//returns the number of pixels drawn
__device__ int rasterizeLine(glm::vec3 start, glm::vec3 finish, fragment* depthBuffer, glm::vec2 resolution, triangle currTri, int triIdx, glm::vec3 lineColor){
	float X, Y, Xinc, Yinc, LENGTH;
	Xinc = finish.x - start.x;
	Yinc = finish.y - start.y;
	int sgnXinc = Xinc > 0 ? 1 : -1;
	int sgnYinc = Yinc > 0 ? 1 : -1;
	int pixelsDrawn = 0;

	glm::vec3 LINE_COLOR(0, 0, 0);
	//if both zero, then we just draw a point.
	if( (abs(Xinc) < NATHANS_EPSILON) && (abs(Yinc) < NATHANS_EPSILON) ){
		writeColorPoint(currTri, triIdx, glm::vec2(start.x, start.y), depthBuffer, resolution, lineColor);
		pixelsDrawn++;
	} else { //this is a line segment
		//LENGTH is the greater of Xinc, Yinc
		if(abs(Yinc) > abs(Xinc)){
			LENGTH = abs(Yinc);
			Xinc = Xinc / LENGTH; //note float division
			Yinc = sgnYinc * 1.0; //step along Y by pixels
		} else {
			LENGTH = abs(Xinc);
			Yinc = Yinc / LENGTH; //note float division
			Xinc = sgnXinc * 1.0; //step along X by pixels
		}
		X = start.x;
		Y = start.y;
		for(int i = 0; i <= roundf(LENGTH); i++){ //do this at least once
			writeColorPoint(currTri, triIdx, glm::vec2(X, Y), depthBuffer, resolution, lineColor);
			pixelsDrawn++;
			X += Xinc;
			Y += Yinc;
		}
	} //end else 'this is a line segment'
	return pixelsDrawn;
}

__global__ void vertexShadeKernel(float* vbo, float* model_vbo, float* nbo, int vbosize, glm::mat4 cameraMat, glm::mat4 model, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){ //each thread acts per vertex.
	  int vertNum = 3*index;
	  glm::vec4 currVert(vbo[vertNum], vbo[vertNum+1], vbo[vertNum+2], 1);
	  glm::vec4 currNorm(nbo[vertNum], nbo[vertNum+1], nbo[vertNum+2], 1);
	  glm::vec4 projectedVert = cameraMat * currVert;
	  projectedVert = (1/projectedVert.w) * projectedVert; //perspective divide
	  float xWinNDC = (projectedVert.x + 1)/2.0f; //shift to window NDC space (between 0 and 1)
	  float yWinNDC = (projectedVert.y + 1)/2.0f; //shift to window NDC space (between 0 and 1)
	  vbo[vertNum] = xWinNDC * resolution.x;
	  vbo[vertNum+1] = yWinNDC * resolution.y;
	  vbo[vertNum+2] = projectedVert.z; //no need to change this when shifting to window NDC space

	  glm::vec4 modelVert = model*currVert;
	  model_vbo[vertNum] = modelVert.x;
	  model_vbo[vertNum+1] = modelVert.y;
	  model_vbo[vertNum+2] = modelVert.z;

	  glm::vec4 modelNorm = model*currNorm;
	  nbo[vertNum] = modelNorm.x;
	  nbo[vertNum+1] = modelNorm.y;
	  nbo[vertNum+2] = modelNorm.z;
  }
}

__global__ void primitiveAssemblyKernel(float* vbo, float* model_vbo, float* nbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){ //one thread per primitive
	  int primNum = 3*index;
	  triangle currTri;
	  int ind0 = ibo[primNum];
	  currTri.p0 = glm::vec3(vbo[3*ind0], vbo[3*ind0 + 1], vbo[3*ind0 + 2]);
	  currTri.modelspace_p0 = glm::vec3(model_vbo[3*ind0], model_vbo[3*ind0 + 1], model_vbo[3*ind0 + 2]);
	  currTri.modelspace_n0 = glm::vec3(nbo[3*ind0], nbo[3*ind0 + 1], nbo[3*ind0 + 2]);
	  currTri.c0 = glm::vec3(cbo[3*ind0], cbo[3*ind0 + 1], cbo[3*ind0 + 2]);
	  int ind1 = ibo[primNum + 1];
	  currTri.p1 = glm::vec3(vbo[3*ind1], vbo[3*ind1 + 1], vbo[3*ind1 + 2]);
	  currTri.modelspace_p1 = glm::vec3(model_vbo[3*ind1], model_vbo[3*ind1 + 1], model_vbo[3*ind1 + 2]);
	  currTri.modelspace_n1 = glm::vec3(nbo[3*ind1], nbo[3*ind1 + 1], nbo[3*ind1 + 2]);
	  currTri.c1 = glm::vec3(cbo[3*ind1], cbo[3*ind1 + 1], cbo[3*ind1 + 2]);
	  int ind2 = ibo[primNum + 2];
	  currTri.p2 = glm::vec3(vbo[3*ind2], vbo[3*ind2 + 1], vbo[3*ind2 + 2]);
	  currTri.modelspace_p2 = glm::vec3(model_vbo[3*ind2], model_vbo[3*ind2 + 1], model_vbo[3*ind2 + 2]);
	  currTri.modelspace_n2 = glm::vec3(nbo[3*ind2], nbo[3*ind2 + 1], nbo[3*ind2 + 2]);
	  currTri.c2 = glm::vec3(cbo[3*ind2], cbo[3*ind2 + 1], cbo[3*ind2 + 2]);
	  primitives[index] = currTri;
  }
}

__global__ void wireRasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, 
	glm::vec3 vdir, bool drawLines, bool interpColors, int* writeCount, bool useLargeStep, bool checkWriteCount, bool backfaceCull){

		int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
			  triangle currTri = primitives[index];
	  glm::vec3 v1 = currTri.p1 - currTri.p0;
	  glm::vec3 v2 = currTri.p2 - currTri.p0;
	  glm::vec3 normal = glm::cross(v1, v2);
	  currTri.n0 = normal;

	  if( backfaceCull && glm::dot(normal, vdir) > 0 ){
		  return; //cull face, it's facing away.
	  }

		glm::vec3 lineColor(0, 1, 0);
	  	rasterizeLine(currTri.p0, currTri.p1, depthbuffer, resolution, currTri, index, lineColor);
		rasterizeLine(currTri.p1, currTri.p2, depthbuffer, resolution, currTri, index, lineColor);
		rasterizeLine(currTri.p2, currTri.p0, depthbuffer, resolution, currTri, index, lineColor);
  }
}

//TODO: Implement a rasterization method, such as scanline.
//NATHAN: at each fragment, calculate the barycentric coordinates, and interpolate position/color. 
//for now the normal can just be the cross product of the vectors that make up the face (flat shading).
//NATHAN: add early-z here.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution, 
	glm::vec3 vdir, bool drawLines, bool interpColors, int* writeCount, bool useLargeStep, bool checkWriteCount, bool backfaceCull){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  //based on notes from here: http://sol.gfxile.net/tri/index.html

	  //add really simple backface culling
	  triangle currTri = primitives[index];
	  glm::vec3 v1 = currTri.p1 - currTri.p0;
	  glm::vec3 v2 = currTri.p2 - currTri.p0;
	  glm::vec3 normal = glm::cross(v1, v2);
	  currTri.n0 = normal;

	  if( backfaceCull && glm::dot(normal, vdir) > 0 ){
		  return; //cull face, it's facing away.
	  }

	  //compute the AABB for the triangle (pad it out so we don't accidentally miss something on the edge)
	  float minX = min(min(currTri.p0.x, currTri.p1.x), currTri.p2.x) - 1;
	  float maxX = max(max(currTri.p0.x, currTri.p1.x), currTri.p2.x) + 1;
	  float minY = min(min(currTri.p0.y, currTri.p1.y), currTri.p2.y) - 1;
	  float maxY = max(max(currTri.p0.y, currTri.p1.y), currTri.p2.y) + 1;

	  //loop through the AABB of the triangle, testing to see if point is in triangle. If yes, write it to depthbuffer. If no, don't.
	  float stepSize;
	  if(useLargeStep)
		  stepSize = 1;
	  else
		  stepSize = 0.5;

	  for(float y = minY; y <= maxY; y = y + stepSize){
		  for(float x = minX; x <= maxX; x = x + stepSize){
			  glm::vec2 currPoint(x, y);
			  glm::vec3 baryCoords = calculateBarycentricCoordinate(currTri, currPoint);
			  if(isBarycentricCoordInBounds(baryCoords)){ //we are inside
				  writePointInTriangle(currTri, index, currPoint, depthbuffer, resolution, interpColors, writeCount, checkWriteCount);
			  }
		  }
	  }

	 // glm::vec3 p0;
	 // glm::vec3 p1;
	 // glm::vec3 p2;
	 // //we want to sort p0, p1, p2, such that p0 has the lowest y-value, p1 middle, p2 highest. 
	 //// There are 6 possible permutations:
	 // if( currTri.p2.y >= currTri.p1.y && currTri.p1.y >= currTri.p0.y ){ //p2 >= p1 >= p0
		//  p2 = currTri.p2; //highest
		//  p1 = currTri.p1; //middle 
		//  p0 = currTri.p0; //low
	 // } else if( currTri.p2.y >= currTri.p0.y && currTri.p0.y >= currTri.p1.y ){ //p2 >= p0 >= p1
		//  p2 = currTri.p2; // highest
		//  p1 = currTri.p0; //middle
		//  p0 = currTri.p1; //low
	 // } else if( currTri.p1.y >= currTri.p2.y && currTri.p2.y >= currTri.p0.y ){ //p1 >= p2 >= p0
		//  p2 = currTri.p1; // highest
		//  p1 = currTri.p2; //middle
		//  p0 = currTri.p0; //low
	 // } else if( currTri.p1.y >= currTri.p0.y && currTri.p0.y >= currTri.p2.y ){ //p1 >= p0 >= p2
		//  p2 = currTri.p1; // highest
		//  p1 = currTri.p0; //middle
		//  p0 = currTri.p2; //low
	 // } else if( currTri.p0.y >= currTri.p2.y && currTri.p2.y >= currTri.p1.y ){ //p0 >= p2 >= p1
		//  p2 = currTri.p0; // highest
		//  p1 = currTri.p2; //middle
		//  p0 = currTri.p1; //low
	 // } else { //p0 >= p1 >= p2
		//  p2 = currTri.p0; // highest
		//  p1 = currTri.p1; //middle
		//  p0 = currTri.p2; //low
	 // }

	  //rasterizeHorizLine(glm::vec2(p1), glm::vec2(p2), depthbuffer, tmp_depthbuffer, resolution, currTri, index);
	  
	  if(drawLines){
		glm::vec3 lineColor(0, 0, 0);
		rasterizeLine(currTri.p0, currTri.p1, depthbuffer, resolution, currTri, index, lineColor);
		rasterizeLine(currTri.p1, currTri.p2, depthbuffer, resolution, currTri, index, lineColor);
		rasterizeLine(currTri.p2, currTri.p0, depthbuffer, resolution, currTri, index, lineColor);
	  }

	  //float d0 = (p1.x - p0.x) / (p1.y - p0.y);
	  //float d1 = (p2.x - p0.x) / (p2.y - p0.y);

	  //float triHeight = (p2.y - p0.y);

	  //if( p0.y > p1.y || p1.y > p2.y){
		 // printf("Trololo\n");
	  //}
	  //TODO: in the size-zero cases, I might have to draw a line.
	  //if( abs(triHeight) > NATHANS_EPSILON ){ //not a size-zero triangle
		 // float topHeight = (p1.y - p0.y);
		 // glm::vec2 gradToMiddle, gradToBottom;
		 // glm::vec2 rasterStart, rasterEnd;
		 // gradToBottom = glm::vec2((p2.x - p0.x) / triHeight, 1);
		 // if( abs(topHeight) > NATHANS_EPSILON ){ //top is not flat
			//  gradToMiddle = glm::vec2((p1.x - p0.x) / topHeight, 1);
			//  rasterStart = glm::vec2(p0);
			//  rasterEnd = glm::vec2(p0);
			//  while(rasterStart.y <= p1.y && rasterEnd.y <= p1.y){
			//	  rasterizeHorizLine(rasterStart, rasterEnd, depthbuffer, resolution, currTri, index);
			//	  rasterStart += gradToBottom;
			//	  rasterEnd += gradToMiddle;
			//  }
			//  rasterStart -= gradToBottom;
			//  rasterEnd = glm::vec2(p1);
		 // } else { //top is flat, thus we don't start at a point, we start at a line
			//  rasterStart = glm::vec2(p0);
			//  rasterEnd = glm::vec2(p1);
			//  rasterizeHorizLine(rasterStart, rasterEnd, depthbuffer, resolution, currTri, index); //this line is the "top"
		 // }
		 // float bottomHeight = (p2.y - p1.y);
		 // if( abs(bottomHeight) > NATHANS_EPSILON ){ //bottom is not flat
			//  glm::vec2 gradMidToBot = glm::vec2((p2.x - p1.x)/bottomHeight, 1);
			//  while(rasterStart.y <= p2.y && rasterEnd.y <= p2.y){
			//	  rasterizeHorizLine(rasterStart, rasterEnd, depthbuffer, resolution, currTri, index);
			//	  rasterStart += gradToBottom;
			//	  rasterEnd += gradMidToBot;
			//  }
		 // } else { //bottom is flat, but we need to rasterize at least one line
			//  rasterizeHorizLine(glm::vec2(p1), glm::vec2(p2), depthbuffer, resolution, currTri, index);
		 // }
	  //} else{ //rasterize two straight lines, since the triangle is "flat"
		 // rasterizeHorizLine(glm::vec2(p0), glm::vec2(p1), depthbuffer, resolution, currTri, index);
		 // rasterizeHorizLine(glm::vec2(p1), glm::vec2(p2), depthbuffer, resolution, currTri, index);
	  //}
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 eyePos, glm::vec3 lightPos, bool useShading, int* write_count, bool checkWriteCount){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x > 0 && y > 0 && x<=resolution.x && y<=resolution.y){
	  fragment currFrag = depthbuffer[index];
		//float depthCoeff = abs(currFrag.position.z) - 0.5f;
		//currFrag.color = currFrag.color * depthCoeff;
		//depthbuffer[index] = currFrag;
	  //float diffuseCoeff = glm::dot(currFrag.normal, glm::normalize(eyePos - currFrag.position));
	  //currFrag.color = currFrag.color * diffuseCoeff;
	  //depthbuffer[index] = currFrag;

	  if( currFrag.triIdx >= 0){
		  if(useShading){
			  //currFrag.color = currFrag.modelNormal;
			  glm::vec3 lightVec = glm::normalize(lightPos - currFrag.modelPosition);
			  float diffuseCoeff = glm::clamp(glm::dot(currFrag.modelNormal, lightVec), 0.0f, 1.0f);
			  currFrag.color = diffuseCoeff * currFrag.color;
		  }

		  if( checkWriteCount && write_count[index] > 1 ){ //yellow indicates overlap!
			  currFrag.color = glm::vec3(1, 1, 0);
		  }

		  depthbuffer[index] = currFrag;
	  }
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x >= 0 && y >= 0 && x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, 
	float angleDeg, glm::vec3 camPos, bool drawLines, bool useShading, bool interpColors, bool useLargeStep, bool checkWriteCount, bool backfaceCull, glm::quat currRot, bool wireframe){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3));
  
  //set up depthbuffer
  depthbuffer = NULL;
  cudaMalloc((void**)&depthbuffer, (int)resolution.x*(int)resolution.y*sizeof(fragment));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(1,1,1));
  
  fragment frag;
  frag.color = glm::vec3(1,1,1);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle));

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int));
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  framebuffer_writes = NULL;
  cudaMalloc((void**)&framebuffer_writes, (int)resolution.x*(int)resolution.y*sizeof(int));
  cudaMemset( framebuffer_writes, 0, (int)resolution.x*(int)resolution.y*sizeof(int));

  //NBO is the size of the VBO
  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, vbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  modelspace_vbo = NULL;
  cudaMalloc((void**)&modelspace_vbo, vbosize*sizeof(float));
  cudaMemcpy( modelspace_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  //hardcoding the camera for now:
  float fovy = 45.0f;
  float zNear = 0.1f;
  float zFar = 100.0f;
  float aspectRatio = resolution.x / resolution.y;
  glm::vec3 up(0,1,0);
  //glm::vec3 up = currUp;
  glm::vec3 center(0,0,0);
  //glm::vec3 eye = camPos;
  glm::vec3 eye(0, 0, 1);
  //glm::vec3 eye = center - currView;
  glm::mat4 projection = glm::perspective(fovy, aspectRatio, zNear, zFar);
  glm::mat4 view = glm::lookAt(eye, center, up);
  //float angleRad = angleDeg * PI/180;
  glm::mat4 trans = glm::translate(glm::mat4(1), -camPos); 
  //model = glm::rotate(model, angleDeg, glm::vec3(0,1,0));
  //glm::mat3 rot = glm::mat3(currRot);
  glm::mat4 model = trans * glm::mat4_cast(glm::inverse(currRot));
  glm::mat4 cameraMat = projection*view*model;
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, modelspace_vbo, device_nbo, vbosize, cameraMat, model, resolution);
  //float* transformedVerts = new float[vbosize];
  //cudaMemcpy( transformedVerts, device_vbo, vbosize*sizeof(float), cudaMemcpyDeviceToHost);
  //delete transformedVerts;

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, modelspace_vbo, device_nbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);
  //triangle* assembledTris = new triangle[ibosize/3];
  //cudaMemcpy( assembledTris, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost);
  //delete assembledTris;

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  //first draw the outlines of the triangle
  glm::vec3 vdir = center - eye;

  if(wireframe){
	  wireRasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, vdir, drawLines, interpColors, framebuffer_writes, useLargeStep, checkWriteCount, backfaceCull);
  } else {
	rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution, vdir, drawLines, interpColors, framebuffer_writes, useLargeStep, checkWriteCount, backfaceCull);
  }
  cudaDeviceSynchronize();
  //next, march through all scanlines
  //int scanlineBlocks = ceil(resolution.y/(float)tileSize);
  //scanlineMarchKernel<<<scanlineBlocks, tileSize>>>(primitives, depthbuffer, tmp_zbuffer, resolution);


  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec3 lightPos(0, 0, 1);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, eye, lightPos, useShading, framebuffer_writes, checkWriteCount);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_nbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( modelspace_vbo );
  cudaFree(framebuffer_writes);
}

