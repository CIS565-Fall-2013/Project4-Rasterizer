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
	std::cin.get ();
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
	  f.position.z = 1e6;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  __shared__ glm::mat4 model;
  __shared__ glm::mat4 view;
  __shared__ glm::mat4 projection;
  __shared__ glm::mat4	ModelViewProjection;
  __shared__ int	step;

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
	  model = glm::mat4 (1.0f);
	  view = glm::mat4 (1.0f);
	  projection = glm::mat4 (1.0f);

	  ModelViewProjection = projection * view * model;
	  step = vbosize/4;
  }

  __syncthreads ();

  if(index<step)
  {
	  glm::vec4 currentVertex (vbo [index], vbo [index+step], vbo [index+(2*step)], vbo [index+(3*step)]);
	  currentVertex = ModelViewProjection * currentVertex;
	  vbo [index] = currentVertex.x;	vbo [index+step] = currentVertex.y;	vbo [index+(2*step)] = currentVertex.z;	vbo [index+(3*step)] = currentVertex.w;
  }
}

//TODO: Implement primitive assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives)
{
  __shared__ int colourStep;
  __shared__ int indexStep;		// = primitivesCount.
  __shared__ int vertStep;
  
  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
	  colourStep = cbosize / 3;
	  vertStep = vbosize/4;
	  indexStep = ibosize / 3;
  }

  __syncthreads ();

  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
//  int primitivesCount = ibosize/3;

  if(index < indexStep/*primitivesCount*/)
  {
	  triangle thisTriangle;
	  
	  int curIndex = ibo [index];
	  thisTriangle.c0.x = cbo [curIndex];	thisTriangle.c0.y = cbo [curIndex + colourStep];	thisTriangle.c0.z = cbo [curIndex + (2*colourStep)];
	  thisTriangle.p0.x = vbo [curIndex];	thisTriangle.p0.y = vbo [curIndex + vertStep];		thisTriangle.p0.z = vbo [curIndex + (2*vertStep)];		thisTriangle.p0.w = vbo [curIndex + (3*vertStep)];

	  curIndex = ibo [index+indexStep];
	  thisTriangle.c1.x = cbo [curIndex];	thisTriangle.c1.y = cbo [curIndex + colourStep];	thisTriangle.c1.z = cbo [curIndex + (2*colourStep)];
	  thisTriangle.p1.x = vbo [curIndex];	thisTriangle.p1.y = vbo [curIndex + vertStep];		thisTriangle.p1.z = vbo [curIndex + (2*vertStep)];		thisTriangle.p1.w = vbo [curIndex + (3*vertStep)];

	  curIndex = ibo [index+(2*indexStep)];
	  thisTriangle.c2.x = cbo [curIndex];	thisTriangle.c2.y = cbo [curIndex + colourStep];	thisTriangle.c2.z = cbo [curIndex + (2*colourStep)];
	  thisTriangle.p2.x = vbo [curIndex];	thisTriangle.p2.y = vbo [curIndex + vertStep];		thisTriangle.p2.z =	vbo [curIndex + (2*vertStep)];		thisTriangle.p2.w = vbo [curIndex + (3*vertStep)];
	  
	  primitives [index] = thisTriangle;
  }
}

// Converts all triangls to screen space.
__global__ void convertToScreenSpace(triangle* primitives, int primitivesCount, glm::vec2 resolution)
{
  extern __shared__ triangle primitiveShared [];
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount)
  {
	  primitiveShared [threadIdx.x] = primitives [index];
	  
	  // Convert clip space coordinates to NDC (a.k.a. Perspective divide).
	  primitiveShared [threadIdx.x].p0.x /= primitiveShared [threadIdx.x].p0.w;
	  primitiveShared [threadIdx.x].p0.y /= primitiveShared [threadIdx.x].p0.w;
	  primitiveShared [threadIdx.x].p0.z /= primitiveShared [threadIdx.x].p0.w;

	  primitiveShared [threadIdx.x].p1.x /= primitiveShared [threadIdx.x].p1.w;
	  primitiveShared [threadIdx.x].p1.y /= primitiveShared [threadIdx.x].p1.w;
	  primitiveShared [threadIdx.x].p1.z /= primitiveShared [threadIdx.x].p1.w;

	  primitiveShared [threadIdx.x].p2.x /= primitiveShared [threadIdx.x].p2.w;
	  primitiveShared [threadIdx.x].p2.y /= primitiveShared [threadIdx.x].p2.w;
	  primitiveShared [threadIdx.x].p2.z /= primitiveShared [threadIdx.x].p2.w;

	  // Rescale NDC to be in the range 0.0 to 1.0.
	  primitiveShared [threadIdx.x].p0.x += 1.0f;
	  primitiveShared [threadIdx.x].p0.x /= 2.0f;
	  primitiveShared [threadIdx.x].p0.y += 1.0f;
	  primitiveShared [threadIdx.x].p0.y /= 2.0f;
	  primitiveShared [threadIdx.x].p0.z += 1.0f;
	  primitiveShared [threadIdx.x].p0.z /= 2.0f;

	  primitiveShared [threadIdx.x].p1.x += 1.0f;
	  primitiveShared [threadIdx.x].p1.x /= 2.0f;
	  primitiveShared [threadIdx.x].p1.y += 1.0f;
	  primitiveShared [threadIdx.x].p1.y /= 2.0f;
	  primitiveShared [threadIdx.x].p1.z += 1.0f;
	  primitiveShared [threadIdx.x].p1.z /= 2.0f;

	  primitiveShared [threadIdx.x].p2.x += 1.0f;
	  primitiveShared [threadIdx.x].p2.x /= 2.0f;
	  primitiveShared [threadIdx.x].p2.y += 1.0f;
	  primitiveShared [threadIdx.x].p2.y /= 2.0f;
	  primitiveShared [threadIdx.x].p2.z += 1.0f;
	  primitiveShared [threadIdx.x].p2.z /= 2.0f;

	  // Now multiply with resolution to get screen co-ordinates.
	  primitiveShared [threadIdx.x].p0.x *= resolution.x;
	  primitiveShared [threadIdx.x].p0.y *= resolution.y;
	  
	  primitiveShared [threadIdx.x].p1.x *= resolution.x;
	  primitiveShared [threadIdx.x].p1.y *= resolution.y;
	  
	  primitiveShared [threadIdx.x].p2.x *= resolution.x;
	  primitiveShared [threadIdx.x].p2.y *= resolution.y;

	  primitives [index] = primitiveShared [threadIdx.x];
  }

//  __syncthreads ();

  //if(index<primitivesCount)
  //{
	 // fragment	curFragment;
	 // curFragment.position.z = 1e6;
	 // // First, throw out all back facing tris (Back Face Culling).
	 // // Here, we simply do nothing if we find such a tri.
	 // if (calculateSignedArea (primitiveShared [threadIdx.x]) > 0)
	 // {
		//  // Next, check if the pixel handled by the current thread is inside the bounding box of tri.
		//  if (
		//  primitiveShared [threadIdx.x].;
	 // }
  //}
}

// Core rasterization.
__global__ void rasterizationKernel (triangle* primitive, fragment* depthbuffer, glm::vec2 resolution)
{
  extern __shared__ fragment zBufferShared [];
  __shared__ triangle	currentPrim;
  __shared__ glm::vec2  bBoxMin;
  __shared__ glm::vec2	bBoxMax;

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
	  currentPrim = *primitive; 
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = (y * resolution.x) + x;

  if (index < (resolution.x*resolution.y))
	  zBufferShared [threadIdx.x] = depthbuffer [index];

  if ((threadIdx.x == 0) && (threadIdx.y == 0))
  {
		// Calculate the bounding box of primitive.
		bBoxMin.x = min (currentPrim.p0.x, min (currentPrim.p1.x, currentPrim.p2.x));
		bBoxMax.x = max (currentPrim.p0.x, max (currentPrim.p1.x, currentPrim.p2.x));

  		bBoxMin.y = min (currentPrim.p0.y, min (currentPrim.p1.y, currentPrim.p2.y));
		bBoxMax.y = max (currentPrim.p0.y, max (currentPrim.p1.y, currentPrim.p2.y));
  }
  __syncthreads ();

  if ((x < resolution.x) && (y < resolution.y))
  {
	  // Throw out current primitive if it's back facing (Back Face Culling).
	  // Here, we simply do nothing.
	  if (calculateSignedArea (currentPrim) > 0)
	  {
		  // Next, check if the pixel handled by the current thread is inside the bounding box of tri.
		  if ((x >= bBoxMin.x) && (x <= bBoxMax.x))
			  if ((y >= bBoxMin.y) && (y <= bBoxMax.y))
			  {
				  fragment	curFragment;
				  glm::vec3 baryCoord = calculateBarycentricCoordinate (currentPrim, glm::vec2 (x,y));
				  // Then, check if the pixel is inside tri.
				  if (isBarycentricCoordInBounds (baryCoord))
				  {  
					  curFragment.color = baryCoord.x * currentPrim.c0 + 
										  baryCoord.y * currentPrim.c1 + 
										  baryCoord.z * currentPrim.c2;

//					  curFragment.normal =	baryCoord.r * currentPrim.c0 + 
//											baryCoord.b * currentPrim.c1 + 
//											baryCoord.g * currentPrim.c2;
					  
					  curFragment.position.x =	baryCoord.x * currentPrim.p0.x + 
												baryCoord.y * currentPrim.p1.x + 
												baryCoord.z * currentPrim.p2.x;

					  curFragment.position.y =	baryCoord.x * currentPrim.p0.y + 
												baryCoord.y * currentPrim.p1.y + 
												baryCoord.z * currentPrim.p2.y;

					  curFragment.position.z =	baryCoord.x * currentPrim.p0.z + 
												baryCoord.y * currentPrim.p1.z + 
												baryCoord.z * currentPrim.p2.z;

					  if (zBufferShared [threadIdx.x].position.z > curFragment.position.z)
						  zBufferShared [threadIdx.x] = curFragment;
				  }
			  }

		  depthbuffer [index] = zBufferShared [threadIdx.x];
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution)
{
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y)
  {
	  ;
  }
}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer)
{

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y)
  {
    framebuffer[index] = depthbuffer[index].color;
  }
}

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize)
{
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
  int primitiveBlocks = ceil(((float)vbosize/4)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize);
  checkCUDAError("Vertex shader failed!");
  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);
  checkCUDAError("Primitive Assembly failed!");
  cudaDeviceSynchronize();

  //------------------------------
  // Map to Screen Space
  //------------------------------
  convertToScreenSpace<<<primitiveBlocks, tileSize, tileSize>>>(primitives, ibosize/3, resolution);
  checkCUDAError("Conversion to Screen Space failed!");
  cudaDeviceSynchronize();

  //-----------------------------------------
  // Rasterization - rasterize each primitive
  //-----------------------------------------
  for (int i = 0; i<(ibosize / 3);  i++)
	  rasterizationKernel<<<fullBlocksPerGrid, threadsPerBlock, threadsPerBlock.x*threadsPerBlock.y>>>>(&primitives[i], depthbuffer, resolution);
  checkCUDAError("Rasterization failed!");
  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);
  checkCUDAError("Fragment shader failed!");
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

void kernelCleanup()
{
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

