// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer;
varying* interpVariables;
float* device_vbo;
float* device_vbo_eye;
float* device_nbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

void checkCUDAError(const char *msg) {
  cudaError_t err = cudaGetLastError();
  if( cudaSuccess != err) {
    fprintf(stderr, "Cuda error: %s: %s.\n", msg, cudaGetErrorString( err) ); 
	getchar();
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

__host__ __device__ void printVec4(glm::vec4 m){
//    std::cout << m[0] << " " << m[1] << " " << m[2] << " " << m[3] << std::endl;
	printf("%f, %f, %f, %f;\n", m[0], m[1], m[2], m[3]);
}

__host__ __device__ void printVec3(glm::vec3 m){
//    std::cout << m[0] << " " << m[1] << " " << m[2] << std::endl;
	printf("%f, %f, %f;\n", m[0], m[1], m[2]);
}


__host__ __device__ glm::vec3 generateRandomNumberFromThread(float time, int index)
{
    thrust::default_random_engine rng(hash(index*time));
    thrust::uniform_real_distribution<float> u01(0,1);

    return glm::vec3((float) u01(rng), (float) u01(rng), (float) u01(rng));
}


//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, varying frag, varying* interpVariables, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    interpVariables[index] = frag;
  }
}

//Reads a fragment from a given location in a fragment buffer
__host__ __device__ varying getFromDepthbuffer(int x, int y, varying* interpVariables, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
    return interpVariables[index];
  }else{
    varying f;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, varying* buffer, varying frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      varying f = frag;
      f.position.x = x;
      f.position.y = y;
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

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(glm::vec2 resolution, glm::mat4 projection, glm::mat4 view, float zNear, float zFar, float* vbo, float *vbo_eye, int vbosize, float *nbo, int nbosize){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  // vertex assembly
	  glm::vec4 vertex(vbo[3*index], vbo[3*index+1], vbo[3*index+2], 1.0f);
	  glm::vec4 normal(nbo[3*index], nbo[3*index+1], nbo[3*index+2], 1.0f);
	  // transform position to eye space
	  vertex = view*vertex;
	  vbo_eye[3*index]   = vertex.x;
	  vbo_eye[3*index+1] = vertex.y;
	  vbo_eye[3*index+2] = vertex.z;
	  // transform normal to eye space
	  normal = view*normal;
	  nbo[3*index]   = normal.x;
	  nbo[3*index+1] = normal.y;
	  nbo[3*index+2] = normal.z;
	  // project to clip space
	  vertex = projection* vertex;
	  // transform to NDC
	  vertex /= vertex.w; // potential division by zero?
	  // viewport transform
	  vbo[3*index]   = resolution.x * 0.5f * (vertex.x + 1.0f);
	  vbo[3*index+1] = resolution.y * 0.5f * (vertex.y + 1.0f);
	  vbo[3*index+2] = (zFar-zNear)*0.5f*vertex.z + (zFar+zNear)*0.5f;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, float *vbo_eye, int vbosize, float *nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  primitives[index].p0.x = vbo[9*index];   primitives[index].p0.y = vbo[9*index+1]; primitives[index].p0.z = vbo[9*index+2];
	  primitives[index].p1.x = vbo[9*index+3]; primitives[index].p1.y = vbo[9*index+4]; primitives[index].p1.z = vbo[9*index+5];
	  primitives[index].p2.x = vbo[9*index+6]; primitives[index].p2.y = vbo[9*index+7]; primitives[index].p2.z = vbo[9*index+8];

	  primitives[index].eyeCoords0.x = vbo_eye[9*index];   primitives[index].eyeCoords0.y = vbo_eye[9*index+1]; primitives[index].eyeCoords0.z = vbo_eye[9*index+2];
	  primitives[index].eyeCoords1.x = vbo_eye[9*index+3]; primitives[index].eyeCoords1.y = vbo_eye[9*index+4]; primitives[index].eyeCoords1.z = vbo_eye[9*index+5];
	  primitives[index].eyeCoords2.x = vbo_eye[9*index+6]; primitives[index].eyeCoords2.y = vbo_eye[9*index+7]; primitives[index].eyeCoords2.z = vbo_eye[9*index+8];

	  primitives[index].eyeNormal0.x = nbo[9*index];   primitives[index].eyeNormal0.y = nbo[9*index+1]; primitives[index].eyeNormal0.z = nbo[9*index+2];
	  primitives[index].eyeNormal1.x = nbo[9*index+3]; primitives[index].eyeNormal1.y = nbo[9*index+4]; primitives[index].eyeNormal1.z = nbo[9*index+5];
	  primitives[index].eyeNormal2.x = nbo[9*index+6]; primitives[index].eyeNormal2.y = nbo[9*index+7]; primitives[index].eyeNormal2.z = nbo[9*index+8];

	  primitives[index].c0.x = cbo[0];		   primitives[index].c0.y = cbo[1];         primitives[index].c0.z = cbo[2];  
	  primitives[index].c1.x = cbo[3];         primitives[index].c1.y = cbo[4];		    primitives[index].c1.z = cbo[5];
	  primitives[index].c2.x = cbo[6];         primitives[index].c2.y = cbo[7];         primitives[index].c2.z = cbo[8];

	  // clipping operation?

  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, varying* interpVariables, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){// any practical use for discarding? introduced divergence and have to wait other thread to finish
	  triangle thisTriangle = primitives[index];	
	  // if degenerate, skip
	  if(abs(calculateSignedArea(thisTriangle)) < 1e-6) return;
	  else
	  {
		  glm::vec3 triMin, triMax;
		  getAABBForTriangle(thisTriangle, triMin, triMax);  
		  // if wholly outside of rendering area, discard
		  if(triMin.x > resolution.x || triMin.y > resolution.y || triMin.z > zFar || 
			 triMax.x < 0            || triMax.y < 0            || triMax.z < zNear) return; // all out-of-screen tris not culled 
		  
		  else {
			  glm::vec2 pixelCoords;
			  glm::vec3 barycentricCoords;
			  int pixelIndex;
			  for(int j = int(triMin.y); j < int(triMax.y+1); ++j)
			  {
				  glm::vec2 Q0(triMin.x, float(j+0.5));
				  glm::vec2 Q1(triMax.x, float(j+0.5));
				  glm::vec2 u = Q1 - Q0;
				  float s[3];
				  float t[3];
				  float minS = 1.0f, maxS = 0.0f;
				  glm::vec2 v0((thisTriangle.p1 - thisTriangle.p0).x, (thisTriangle.p1 - thisTriangle.p0).y);
				  glm::vec2 v1((thisTriangle.p2 - thisTriangle.p1).x, (thisTriangle.p2 - thisTriangle.p1).y);
				  glm::vec2 v2((thisTriangle.p0 - thisTriangle.p2).x, (thisTriangle.p0 - thisTriangle.p2).y);

				  glm::vec2 w;
				  if(abs(u.x*v0.y - u.y*v0.x) > 1e-6)
				  {
					  w = Q0 - glm::vec2(thisTriangle.p0.x, thisTriangle.p0.y);
					  s[0] = (v0.y*w.x - v0.x*w.y) / (v0.x*u.y - v0.y*u.x);
					  t[0] = (u.x*w.y  - u.y*w.x ) / (u.x*v0.y - u.y*v0.x);
					  if(s[0] > 0 && s[0] < 1 && t[0] > 0 && t[0] < 1)
					  {
						  minS = fminf(s[0], minS);
						  maxS = fmaxf(s[0], maxS);
					  }
				  }
				  if(abs(u.x*v1.y - u.y*v1.x) > 1e-6)
				  {
					  w = Q0 - glm::vec2(thisTriangle.p1.x, thisTriangle.p1.y);
					  s[1] = (v1.y*w.x - v1.x*w.y) / (v1.x*u.y - v1.y*u.x);
					  t[1] = (u.x*w.y  - u.y*w.x ) / (u.x*v1.y - u.y*v1.x);
					  if(s[1] > 0 && s[1] < 1 && t[1] > 0 && t[1] < 1)
					  {
						  minS = fminf(s[1], minS);
						  maxS = fmaxf(s[1], maxS);
					  }
				  }
				  if(abs(u.x*v2.y - u.y*v2.x) > 1e-6)
				  {
					  w = Q0 - glm::vec2(thisTriangle.p2.x, thisTriangle.p2.y);
					  s[2] = (v2.y*w.x - v2.x*w.y) / (v2.x*u.y - v2.y*u.x);
					  t[2] = (u.x*w.y  - u.y*w.x ) / (u.x*v2.y - u.y*v2.x);
					  if(s[2] > 0 && s[2] < 1 && t[2] > 0 && t[2] < 1)
					  {
						  minS = fminf(s[2], minS);
						  maxS = fmaxf(s[2], maxS);
					  }
				  }
				  for(int i = int(triMin.x + minS * u.x); i < int(triMin.x + maxS * u.x + 1); ++i)
				  {
					  pixelCoords = glm::vec2(float(i+0.5), float(j+0.5));
					  barycentricCoords = calculateBarycentricCoordinate(thisTriangle, pixelCoords);
					  pixelIndex = resolution.x - 1 - i + ((resolution .y  - 1 - j) * resolution.x);
					  if(isBarycentricCoordInBounds(barycentricCoords))
					  {
					  	  interpVariables[pixelIndex].position = barycentricCoords.x * thisTriangle.eyeCoords0 + barycentricCoords.y * thisTriangle.eyeCoords1 + barycentricCoords.z * thisTriangle.eyeCoords2;						  
					  	  interpVariables[pixelIndex].normal   = barycentricCoords.x * thisTriangle.eyeNormal0 + barycentricCoords.y * thisTriangle.eyeNormal1 + barycentricCoords.z * thisTriangle.eyeNormal2; 
						  interpVariables[pixelIndex].color    = barycentricCoords.x * thisTriangle.c0         + barycentricCoords.y * thisTriangle.c1         + barycentricCoords.z * thisTriangle.c2; 
					  }	
				  }
			  }
		  }
	  }
   }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(varying* interpVariables, glm::vec2 resolution){// input: position, normal, intrinsic color, light position, eye position; output: true color
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
//	  interpVariables[index].color = generateRandomNumberFromThread(1, index);
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, varying* interpVariables, glm::vec3* framebuffer){// write true color to framebuffer

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = interpVariables[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, glm::mat4 projection, glm::mat4 view, float zNear, float zFar, float* vbo, int vbosize, float *nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3)); // frame buffer store colors
  
  //set up interpVariables
  interpVariables = NULL;
  cudaMalloc((void**)&interpVariables, (int)resolution.x*(int)resolution.y*sizeof(varying)); // interpolation result per pixel by rasterizer

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0)); // launch kernel for every pixel
  
  varying frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(0,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, interpVariables,frag); // launch kernel for every pixel

  //------------------------------
  //memory stuff
  //------------------------------
  primitives = NULL;
  cudaMalloc((void**)&primitives, (ibosize/3)*sizeof(triangle)); // store all triangles/primitives

  device_ibo = NULL;
  cudaMalloc((void**)&device_ibo, ibosize*sizeof(int)); // vbosize == number of vertices
  cudaMemcpy( device_ibo, ibo, ibosize*sizeof(int), cudaMemcpyHostToDevice);

  device_vbo = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float)); // vbosize == number of vertex components
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_vbo_eye = NULL;
  cudaMalloc((void**)&device_vbo_eye, vbosize*sizeof(float)); // vbosize == number of vertex components

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float)); // nbosize == number of normal components
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float)); // cbosize == 9
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize)); // launch for every vertex

  //------------------------------
  //vertex shader
  //------------------------------
//  cudaMat4 cuProjection = utilityCore::glmMat4ToCudaMat4(projection);
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(resolution, projection, view, zNear, zFar, device_vbo, device_vbo_eye, vbosize, device_nbo, nbosize);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize)); // launch for every primitive
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_eye, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, interpVariables, resolution, zNear, zFar); // launch for every primitive
  checkCUDAError("Kernel failed!");
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(interpVariables, resolution); // launch for every pixel

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, interpVariables, framebuffer); // launch for every pixel
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer); // launch for every pixel

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_vbo_eye );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( interpVariables );
}

