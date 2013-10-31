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
fragment* depthbuffer;
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

//Writes a given fragment to a fragment buffer at a given location
__host__ __device__ void writeToDepthbuffer(int x, int y, fragment frag, fragment* depthbuffer, glm::vec2 resolution){
  if(x<resolution.x && y<resolution.y){
    int index = (y*resolution.x) + x;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, fragment* buffer, fragment frag){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      fragment f = frag;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 MVP, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
#if THREE == 0 
  int size = vbosize / 4;
#else 
  int size = vbosize / 3;
#endif

  if(index < size){

#if THREE == 0
	  glm::vec4 in_vertex = glm::vec4(vbo[4 * index], vbo[4 * index + 1], vbo[4 * index + 2], vbo[4 * index + 3]);
#else 
	  glm::vec4 in_vertex = glm::vec4(vbo[3 * index], vbo[3 * index + 1], vbo[3 * index + 2], 1.0f);
#endif
	  
	  glm::vec4 P = MVP * in_vertex;

#if POINT_MODE == 1
	  glm::vec3 P_ndc = glm::vec3(P.x / P.w, P.y / P.w, P.z / P.w);

	  P.x = resolution.x / 2 * P_ndc.x + resolution.x / 2;
	  P.y = resolution.y / 2 * P_ndc.y + resolution.y / 2;
	  P.z = P_ndc.z;
#endif
	  
#if THREE == 0
	  vbo[4 * index] = P.x;
	  vbo[4 * index + 1] = P.y;
	  vbo[4 * index + 2] = P.z;
	  vbo[4 * index + 3] = P.w;
#else
	  vbo[3 * index] = P.x;
	  vbo[3 * index + 1] = P.y
	  vbo[3 * index + 2] = P.z;
#endif

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index < primitivesCount){
	  int a = ibo[3 * index], b = ibo[3 * index + 1], c = ibo[3 * index + 2];
	
	  
	  // Create triangle
	  triangle t;
#if THREE == 0
	  t.p0 = glm::vec4(vbo[4 * a], vbo[4 * a + 1], vbo[4 * a + 2], vbo[4 * a + 3]);
	  t.p1 = glm::vec4(vbo[4 * b], vbo[4 * b + 1], vbo[4 * b + 2], vbo[4 * b + 3]);
	  t.p2 = glm::vec4(vbo[4 * c], vbo[4 * c + 1], vbo[4 * c + 2], vbo[4 * c + 3]);
#else 
	  t.p0 = glm::vec3(vbo[3 * a], vbo[3 * a + 1], vbo[3 * a + 2]);
	  t.p1 = glm::vec3(vbo[3 * b], vbo[3 * b + 1], vbo[3 * b + 2]);
	  t.p2 = glm::vec3(vbo[3 * c], vbo[3 * c + 1], vbo[3 * c + 2]);
#endif

	  t.c0 = t.c1 = t.c2 = glm::vec3(1.0f);

	  // TODO : Cull Back-Face Triangle

	  // TODO : Cull Out-of-Viewport Triangles

	  // Put triangle into primitives
	  primitives[index] = t;
  }
}

__global__ void viewportTransformKernel(triangle* primitives, int primitivesCount, glm::vec2 resolution){
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index < primitivesCount){
		triangle t = primitives[index];

		// Transform to NDC
		glm::vec4 p0_ndc = t.p0 / t.p0.w;
		glm::vec4 p1_ndc = t.p1 / t.p1.w;
		glm::vec4 p2_ndc = t.p2 / t.p2.w;

		// Transform to Screen Coords
		t.p0.x = 1.0f / 2 * resolution.x * p0_ndc.x + (1.0f * 2 * resolution.x);
		t.p0.y = 1.0f / 2 * resolution.y * p0_ndc.y + (1.0f * 2 * resolution.y);
		t.p0.z = p0_ndc.z;

		t.p1.x = 1.0f / 2 * resolution.x * p1_ndc.x + (1.0f * 2 * resolution.x);
		t.p1.y = 1.0f / 2 * resolution.y * p1_ndc.y + (1.0f * 2 * resolution.y);
		t.p1.z = p0_ndc.z;

		t.p2.x = 1.0f / 2 * resolution.x * p2_ndc.x + (1.0f * 2 * resolution.x);
		t.p2.y = 1.0f / 2 * resolution.y * p2_ndc.y + (1.0f * 2 * resolution.y);
		t.p2.z = p0_ndc.z;
	}
}

__global__ void rasterizePoints(float* vbo, float vbosize, fragment* depthbuffer, glm::vec2 resolution)
{
  int index = threadIdx.x + (blockDim.x * blockIdx.x);

#if THREE == 0
  int size = vbosize / 4;
#else 
  int size = vbosize / 3;
#endif

  if(index < size){

#if THREE == 0
	  glm::vec2 point = glm::vec2(vbo[4 * index], vbo[4 * index + 1]);
#else
	  glm::vec2 point = glm::vec2(vbo[3 * index], vbo[3 * index + 1]);
#endif

	  if(point.x >=0 && point.x < resolution.x && point.y >=0 && point.y < resolution.y)
	  {
		  fragment f;
		  f.position.x = point.x;
		  f.position.y = point.y;

#if THREE == 0
		  f.position.z = vbo[4 * index + 2];
#else 
		  f.position.z = vbo[3 * index + 2];
#endif

		  int bufferindex = int(point.x) + int(point.y) * resolution.x;
		  f.color = glm::vec3(1,1,1);
		  depthbuffer[bufferindex] = f;
	  }
  }
}

// Rasterizer
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  //RASTERIZE(primitives, primitivesCount, depthbuffer, resolution);
  }
}

// Fragment Shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
#if SHADING == 0
	  // FLAT SHADING
	  // no change to color
#elif SHADING == 1
	  // LAMBERT SHADING
	  f.color = f.color * glm::dot(f.normal, glm::vec3(1.0f) - f.position);
#elif SHADING == 2
	  // BLINN-PHONG SHADING
	  glm::vec3 diffuse = f.color * glm::dot(f.normal, LIGHT - f.position);
	  glm::vec3 specular = glm::dot(reflect(LIGHT - f.position, f.normal), -1.0f * f.position) * glm::vec3(1.0f);
	  f.color = .5f * diffuse + .5f * specular;
#endif
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, glm::mat4 MVP, float zNear, float zFar){

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
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0));
  
  fragment frag;
  frag.color = glm::vec3(0,0,0);
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

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, MVP, resolution, zNear, zFar);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
#if POINT_MODE == 0
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //viewport transformation
  //------------------------------
  viewportTransformKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, resolution);

  //------------------------------
  //rasterization
  //------------------------------

  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
#else
  primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));
  rasterizePoints<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, depthbuffer, resolution);
#endif
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  //fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);

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
}

