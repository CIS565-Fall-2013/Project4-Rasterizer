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
float* tmp_zbuffer;
float* device_vbo;
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

//Writes a given fragment to a fragment buffer at a given location
__device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution, float* tmp_depthbuffer){
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
		depthbuffer[index] = frag;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag, float* tmp_buffer){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
      f.position.x = x;
      f.position.y = y;
	  f.triIdx = -1; //no triangle associated.
      buffer[index] = f;
	  tmp_buffer[index] = f.position.z;
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
__device__ void writePointInTriangle(triangle currTri, int triIdx, glm::vec2 xyCoords, fragment* depthBuffer, float* tmp_depthBuffer, glm::vec2 resolution){
	fragment currFrag;
	currFrag.triIdx = triIdx;
	currFrag.color = currTri.c0; //assume the tri is all one color for now.
	glm::vec3 currBaryCoords = calculateBarycentricCoordinate(currTri, xyCoords);
	float fragZ = getZAtCoordinate(currBaryCoords, currTri);
	currFrag.position = glm::vec3(xyCoords.x, xyCoords.y, fragZ);
		//printf("Frag z: %f\n", currFrag.position.z);
	int pixX = roundf(xyCoords.x);
	int pixY = roundf(xyCoords.y);
	//TODO: incorporate the normal in here **somewhere**
	writeToDepthbuffer((resolution.x - 1) - pixX, (resolution.y - 1) - pixY, currFrag, depthBuffer, resolution, tmp_depthBuffer);
	//writeToDepthbuffer(pixX, pixY, currFrag, depthBuffer, resolution);
}

//rasterize between startX and endX, inclusive
__device__ int rasterizeHorizLine(glm::vec2 start, glm::vec2 end, fragment* depthBuffer, float* tmp_depthBuffer, glm::vec2 resolution, triangle currTri, int triIdx){
	int Xinc = roundf(end.x) - roundf(start.x);
	int sgnXinc = Xinc > 0 ? 1 : -1;
	int numPixels = abs(Xinc) + 1; //+1 to be inclusive
	int currX = roundf(start.x);
	int Y = roundf(start.y); //Y should be the same for the whole line
	for(int i = 0; i < numPixels; i++){
		writePointInTriangle(currTri, triIdx, glm::vec2(currX, Y), depthBuffer, tmp_depthBuffer, resolution);
		currX += sgnXinc; //either increase or decrease currX depending on direction
	}
}

//Based on slide 75-76 of the CIS560 notes, Norman I. Badler, University of Pennsylvania. 
//returns the number of pixels drawn
__device__ int rasterizeLine(glm::vec3 start, glm::vec3 finish, fragment* depthBuffer, float* tmp_depthBuffer, glm::vec2 resolution, triangle currTri, int triIdx){
	float X, Y, Xinc, Yinc, LENGTH;
	Xinc = finish.x - start.x;
	Yinc = finish.y - start.y;
	int sgnXinc = Xinc > 0 ? 1 : -1;
	int sgnYinc = Yinc > 0 ? 1 : -1;
	int pixelsDrawn = 0;
	//if both zero, then we just draw a point.
	if( (abs(Xinc) < NATHANS_EPSILON) && (abs(Yinc) < NATHANS_EPSILON) ){
		writePointInTriangle(currTri, triIdx, glm::vec2(start.x, start.y), depthBuffer, tmp_depthBuffer, resolution);
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
			writePointInTriangle(currTri, triIdx, glm::vec2(X, Y), depthBuffer, tmp_depthBuffer, resolution);
			pixelsDrawn++;
			X += Xinc;
			Y += Yinc;
		}
	} //end else 'this is a line segment'
	return pixelsDrawn;
}

__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 cameraMat, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){ //each thread acts per vertex.
	  int vertNum = 3*index;
	  glm::vec4 currVert(vbo[vertNum], vbo[vertNum+1], vbo[vertNum+2], 1);
	  glm::vec4 projectedVert = cameraMat * currVert;
	  projectedVert = (1/projectedVert.w) * projectedVert; //perspective divide
	  float xWinNDC = (projectedVert.x + 1)/2.0f; //shift to window NDC space (between 0 and 1)
	  float yWinNDC = (projectedVert.y + 1)/2.0f; //shift to window NDC space (between 0 and 1)
	  vbo[vertNum] = xWinNDC * resolution.x;
	  vbo[vertNum+1] = yWinNDC * resolution.y;
	  vbo[vertNum+2] = projectedVert.z; //no need to change this when shifting to window NDC space
  }
}

__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){ //one thread per primitive
	  int primNum = 3*index;
	  triangle currTri;
	  int ind0 = ibo[primNum];
	  currTri.p0 = glm::vec3(vbo[3*ind0], vbo[3*ind0 + 1], vbo[3*ind0 + 2]);
	  currTri.c0 = glm::vec3(cbo[3*ind0], cbo[3*ind0 + 1], cbo[3*ind0 + 2]);
	  int ind1 = ibo[primNum + 1];
	  currTri.p1 = glm::vec3(vbo[3*ind1], vbo[3*ind1 + 1], vbo[3*ind1 + 2]);
	  currTri.c1 = glm::vec3(cbo[3*ind1], cbo[3*ind1 + 1], cbo[3*ind1 + 2]);
	  int ind2 = ibo[primNum + 2];
	  currTri.p2 = glm::vec3(vbo[3*ind2], vbo[3*ind2 + 1], vbo[3*ind2 + 2]);
	  currTri.c2 = glm::vec3(cbo[3*ind2], cbo[3*ind2 + 1], cbo[3*ind2 + 2]);
	  primitives[index] = currTri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
//NATHAN: at each fragment, calculate the barycentric coordinates, and interpolate position/color. 
//for now the normal can just be the cross product of the vectors that make up the face (flat shading).
//NATHAN: add early-z here.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, float* tmp_depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  //based on notes from here: http://sol.gfxile.net/tri/index.html
	  triangle currTri = primitives[index];

	  glm::vec3 p0;
	  glm::vec3 p1;
	  glm::vec3 p2;
	  //we want to sort p0, p1, p2, such that p0 has the lowest y-value, p1 middle, p2 highest. 
	 // There are 6 possible permutations:
	  if( currTri.p2.y >= currTri.p1.y && currTri.p1.y >= currTri.p0.y ){ //p2 >= p1 >= p0
		  p2 = currTri.p2; //highest
		  p1 = currTri.p1; //middle 
		  p0 = currTri.p0; //low
	  } else if( currTri.p2.y >= currTri.p0.y && currTri.p0.y >= currTri.p1.y ){ //p2 >= p0 >= p1
		  p2 = currTri.p2; // highest
		  p1 = currTri.p0; //middle
		  p0 = currTri.p1; //low
	  } else if( currTri.p1.y >= currTri.p2.y && currTri.p2.y >= currTri.p0.y ){ //p1 >= p2 >= p0
		  p2 = currTri.p1; // highest
		  p1 = currTri.p2; //middle
		  p0 = currTri.p0; //low
	  } else if( currTri.p1.y >= currTri.p0.y && currTri.p0.y >= currTri.p2.y ){ //p1 >= p0 >= p2
		  p2 = currTri.p1; // highest
		  p1 = currTri.p0; //middle
		  p0 = currTri.p2; //low
	  } else if( currTri.p0.y >= currTri.p2.y && currTri.p2.y >= currTri.p1.y ){ //p0 >= p2 >= p1
		  p2 = currTri.p0; // highest
		  p1 = currTri.p2; //middle
		  p0 = currTri.p1; //low
	  } else { //p0 >= p1 >= p2
		  p2 = currTri.p0; // highest
		  p1 = currTri.p1; //middle
		  p0 = currTri.p2; //low
	  }

	  //rasterizeHorizLine(glm::vec2(p1), glm::vec2(p2), depthbuffer, tmp_depthbuffer, resolution, currTri, index);
	  
	  int numPixels = rasterizeLine(currTri.p0, currTri.p1, depthbuffer, tmp_depthbuffer, resolution, currTri, index);
	  numPixels += rasterizeLine(currTri.p1, currTri.p2, depthbuffer, tmp_depthbuffer, resolution, currTri, index);
	  numPixels += rasterizeLine(currTri.p2, currTri.p0, depthbuffer, tmp_depthbuffer, resolution, currTri, index);
	  //float d0 = (p1.x - p0.x) / (p1.y - p0.y);
	  //float d1 = (p2.x - p0.x) / (p2.y - p0.y);

	  float triHeight = (p2.y - p0.y);

	  if( p0.y > p1.y || p1.y > p2.y){
		  printf("Trololo\n");
	  }
	  //TODO: in the size-zero cases, I might have to draw a line.
	  if( abs(triHeight) > NATHANS_EPSILON ){ //not a size-zero triangle
		  float topHeight = (p1.y - p0.y);
		  glm::vec2 gradToMiddle, gradToBottom;
		  glm::vec2 rasterStart, rasterEnd;
		  gradToBottom = glm::vec2((p2.x - p0.x) / triHeight, 1);
		  if( abs(topHeight) > NATHANS_EPSILON ){ //top is not flat
			  gradToMiddle = glm::vec2((p1.x - p0.x) / topHeight, 1);
			  rasterStart = glm::vec2(p0);
			  rasterEnd = glm::vec2(p0);
			  while(rasterStart.y <= p1.y && rasterEnd.y <= p1.y){
				  rasterizeHorizLine(rasterStart, rasterEnd, depthbuffer, tmp_depthbuffer, resolution, currTri, index);
				  rasterStart += gradToBottom;
				  rasterEnd += gradToMiddle;
			  }
			  rasterStart -= gradToBottom;
			  rasterEnd = glm::vec2(p1);
		  } else { //top is flat, thus we don't start at a point, we start at a line
			  rasterStart = glm::vec2(p0);
			  rasterEnd = glm::vec2(p1);
		  }
		  float bottomHeight = (p2.y - p1.y);
		  if( abs(bottomHeight) > NATHANS_EPSILON ){ //bottom is not flat
			  glm::vec2 gradMidToBot = glm::vec2((p2.x - p1.x)/bottomHeight, 1);
			  while(rasterStart.y <= p2.y && rasterEnd.y <= p2.y){
				  rasterizeHorizLine(rasterStart, rasterEnd, depthbuffer, tmp_depthbuffer, resolution, currTri, index);
				  rasterStart += gradToBottom;
				  rasterEnd += gradMidToBot;
			  }
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x > 0 && y > 0 && x<=resolution.x && y<=resolution.y){
	  fragment currFrag = depthbuffer[index];
	  currFrag.color = currFrag.color * (-0.801f - currFrag.position.z) * 100.0f;
	  //printf("Frag z: %f\n", currFrag.position.z);
	  depthbuffer[index] = currFrag;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float angleDeg, glm::vec3 camPos){

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

  tmp_zbuffer = NULL;
  cudaMalloc((void**)&tmp_zbuffer, (int)resolution.x*(int)resolution.y*sizeof(float));

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag, tmp_zbuffer);

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
  glm::vec3 center(0,0,0);
  //glm::vec3 eye = camPos;
  glm::vec3 eye(0, 0, 1);
  glm::mat4 projection = glm::perspective(fovy, aspectRatio, zNear, zFar);
  glm::mat4 view = glm::lookAt(eye, center, up);
  //float angleRad = angleDeg * PI/180;
  glm::mat4 model = glm::translate(glm::mat4(1), -camPos); 
  model = glm::rotate(model, angleDeg, glm::vec3(0,1,0));
  glm::mat4 cameraMat = projection*view*model;
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, cameraMat, resolution);
  //float* transformedVerts = new float[vbosize];
  //cudaMemcpy( transformedVerts, device_vbo, vbosize*sizeof(float), cudaMemcpyDeviceToHost);
  //delete transformedVerts;

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);
  //triangle* assembledTris = new triangle[ibosize/3];
  //cudaMemcpy( assembledTris, primitives, (ibosize/3)*sizeof(triangle), cudaMemcpyDeviceToHost);
  //delete assembledTris;

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  //first draw the outlines of the triangle
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, tmp_zbuffer, resolution);
  cudaDeviceSynchronize();
  //next, march through all scanlines
  //int scanlineBlocks = ceil(resolution.y/(float)tileSize);
  //scanlineMarchKernel<<<scanlineBlocks, tileSize>>>(primitives, depthbuffer, tmp_zbuffer, resolution);


  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);

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
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
  cudaFree( tmp_zbuffer );
}

