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
//Transform incoming vertex position from model to clip coordinates.
//Use model matrix to transform into model space. Use view matrix to transform into camera space. Then,
//use projection matrix to transform into clip space. Next, convert to NDC. Lastly, convert to window coordinates
__global__ void vertexShadeKernel(float* vbo, int vbosize, mat4 mvp, vec2 reso, mat4 viewport)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;

  if(index<vbosize/3)
  {
	  const int vboId1 = index * 3;
	  const int vboId2 = vboId1 + 1;
	  const int vboId3 = vboId2 + 2;

	  vec4 hPoint = vec4(vbo[vboId1], vbo[vboId2], vbo[vboId3], 1.0);
	  
	  // to clip
	  hPoint = mvp * hPoint;
	  float wClip = hPoint.w;

	  // to ndc
	  vec4 ndcPoint = vec4(hPoint.x / wClip, hPoint.y / wClip, hPoint.z / wClip, hPoint.w / wClip);
	  
	  // to window
	  vec4 windowPoint = viewport * ndcPoint;
	 
	  vbo[vboId1] = windowPoint.x;
	  vbo[vboId2] = windowPoint.y;
	  vbo[vboId3] = windowPoint.z;
  }
}

//TODO: Implement primative assembly
//Given the vbo, cbo, ibo, group them into triangles and output the result
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;

  if(index<primitivesCount)
  {
	  // ibo indices
	  const int iboId1 = index * 3;
	  const int iboId2 = iboId1 + 1;
	  const int iboId3 = iboId1 + 2;

	  // vbo indices for each ibo index
	  const int vboId11 = iboId1;
	  const int vboId12 = iboId1 + 1;
	  const int vboId13 = iboId1 + 2;

	  const int vboId21 = iboId2;
	  const int vboId22 = iboId2 + 1;
	  const int vboId23 = iboId2 + 2;

	  const int vboId31 = iboId3;
	  const int vboId32 = iboId3 + 1;
	  const int vboId33 = iboId3 + 2;
	  
	  // cbo indices
	  const int cboId1 = index % 3;
	  const int cboId2 = cboId1 + 1;
	  const int cboId3 = cboId2 + 2;

	  // retrieve vertices
	  vec3 vert1 = vec3(vbo[vboId11], vbo[vboId12], vbo[vboId13]);
	  vec3 vert2 = vec3(vbo[vboId21], vbo[vboId22], vbo[vboId23]);
	  vec3 vert3 = vec3(vbo[vboId31], vbo[vboId32], vbo[vboId33]);

	  // retrieve colors
	  vec3 vert1Color = vec3(cbo[cboId1], cbo[cboId2], cbo[cboId3]);
	  vec3 vert2Color = vec3(cbo[cboId1], cbo[cboId2], cbo[cboId3]);
	  vec3 vert3Color = vec3(cbo[cboId1], cbo[cboId2], cbo[cboId3]);

	  // build triangle
	  triangle tri;

	  tri.p0 = vert1;
	  tri.p1 = vert2;
	  tri.p2 = vert3;
	  tri.c0 = vert1Color;
	  tri.c1 = vert2Color;
	  tri.c2 = vert3Color;

	  primitives[index] = tri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<primitivesCount)
	{
		triangle tri = primitives[index];
		vec3 triMinPoint;
		vec3 triMaxPoint;

		getAABBForTriangle(tri, triMinPoint, triMaxPoint);
	  
		triMinPoint.x > 0 ? triMinPoint.x : 0;
		triMinPoint.y > 0 ? triMinPoint.y : 0;
		triMaxPoint.x < resolution.x ? triMaxPoint.x : resolution.x;
		triMaxPoint.y < resolution.y ? triMaxPoint.y : resolution.y;

		// go through each pixel within the AABB for the triangle and fill the depthbuffer (fragments) appropriately
		for (int x = triMinPoint.x ; x < triMaxPoint.x ; ++x)
		{
			for (int y = triMinPoint.y ; y < triMaxPoint.y ; ++y)
			{
				vec2 pointInTri(x,y);
				vec3 bc = calculateBarycentricCoordinate(tri, pointInTri);
				if (isBarycentricCoordInBounds(bc))
				{
					// TODO: Compute normal & depth check
					int depthBufferId = x * resolution.x + y;
					float z = getZAtCoordinate(bc, tri);
					depthbuffer[depthBufferId].position = vec3(x, y, z);
					depthbuffer[depthBufferId].color = vec3(tri.c0 * bc.x, tri.c1 * bc.y, tri.c2 * bc.z);
				}
			}
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
void cudaRasterizeCore(camera* cam, uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize)
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
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------

  // copy over camera information
  mat4 modelMatrix(1);
  mat4 viewMatrix = cam->view;
  mat4 projectionMatrix = cam->projection;
  mat4 mvp = projectionMatrix * viewMatrix * modelMatrix;
  vec2 reso = cam->resolution;
  mat4 viewport = cam->viewport;

  // launch vertex shader kernel
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, mvp, reso, viewport);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

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
}

