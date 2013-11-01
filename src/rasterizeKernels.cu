// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "glm/gtc/matrix_transform.hpp"
#include <time.h>

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
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 MVP, glm::mat4 MV, float* vbo2){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  int i0 = 3 * index;
	  int i1 = i0 + 1;
	  int i2 = i0 + 2;
	  glm::vec4 newVbo = MVP * glm::vec4(vbo[i0], vbo[i1], vbo[i2], 1.0);
	  glm::vec4 newVbo2 = MV * glm::vec4(vbo[i0], vbo[i1], vbo[i2], 1.0);
	  vbo[i0] = newVbo.x / newVbo.w;
	  vbo[i1] = newVbo.y / newVbo.w;
	  vbo[i2] = newVbo.z / newVbo.w;
	  vbo2[i0] = newVbo2.x;
	  vbo2[i1] = newVbo2.y;
	  vbo2[i2] = newVbo2.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, float* vbo2){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int i0_x = 3 * ibo[index * 3];
	  int i0_y = i0_x + 1;
	  int i0_z = i0_x + 2;
	  int i1_x = 3 * ibo[index * 3 + 1];
	  int i1_y = i1_x + 1;
	  int i1_z = i1_x + 2;
	  int i2_x = 3 * ibo[index * 3 + 2];
	  int i2_y = i2_x + 1;
	  int i2_z = i2_x + 2;

	  triangle tri;
	  if (i0_z < cbosize)
	  {
		tri.c0 = glm::vec3(cbo[i0_x], cbo[i0_y], cbo[i0_z]);
	  }
	  else
	  {
		tri.c0 = glm::vec3(1.0);
	  }
	  if (i1_z < cbosize)
	  {
		tri.c1 = glm::vec3(cbo[i1_x], cbo[i1_y], cbo[i1_z]);
	  }
	  else
	  {
		tri.c1 = glm::vec3(1.0);
	  }
	  if (i2_z < cbosize)
	  {
	    tri.c2 = glm::vec3(cbo[i2_x], cbo[i2_y], cbo[i2_z]);
	  }
	  else
	  {
		tri.c2 = glm::vec3(1.0);
	  }
	  tri.p0 = glm::vec3(vbo[i0_x], vbo[i0_y], vbo[i0_z]);
	  tri.p1 = glm::vec3(vbo[i1_x], vbo[i1_y], vbo[i1_z]);
	  tri.p2 = glm::vec3(vbo[i2_x], vbo[i2_y], vbo[i2_z]);
	  tri.q0 = glm::vec3(vbo2[i0_x], vbo2[i0_y], vbo2[i0_z]);
	  tri.q1 = glm::vec3(vbo2[i1_x], vbo2[i1_y], vbo2[i1_z]);
	  tri.q2 = glm::vec3(vbo2[i2_x], vbo2[i2_y], vbo2[i2_z]);

	  primitives[index] = tri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){

	int width = resolution.x;
	int height = resolution.y;
	triangle tri = primitives[index];

	glm::vec3 minpoint;
	glm::vec3 maxpoint;

	glm::vec3 v01 = tri.p1 - tri.p0;
	glm::vec3 v12 = tri.p2 - tri.p1;

	float crossProduct = v01.x * v12.y - v01.y * v12.x;

	if (crossProduct < 0) return;

	getAABBForTriangle(tri, minpoint, maxpoint);

	// Change to screen coordinate
	int minX = (minpoint.x + 1) / 2 * width;
	int minY = (minpoint.y + 1) / 2 * height;
	int maxX = (maxpoint.x + 1) / 2 * width + 1;
	int maxY = (maxpoint.y + 1) / 2 * height + 1;

	for (int i=minY; i < maxY; i++)
	{
		float y = (float)i/(float)height * 2.0 - 1.0;

		for (int j=minX; j < maxX; j++)
		{
			float x = (float)j/(float)width * 2.0 - 1.0;
			
			glm::vec2 point(x, y);
			glm::vec3 barycentricCoord = calculateBarycentricCoordinate(tri, point);
			bool isInBounds = isBarycentricCoordInBounds(barycentricCoord);

			if (isInBounds)
			{
				float z = getZAtCoordinate(barycentricCoord, tri);

				fragment frag;
				frag.color = getColorAtCoordinate(barycentricCoord, tri);
				frag.normal = glm::normalize(glm::cross(tri.q1-tri.q0, tri.q2-tri.q1));
				frag.position = getPositionAtCoordinate(barycentricCoord, tri);
				depthbuffer[(height-1-i)*width+(width-1-j)] = frag;
			}
		}
	}
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPos){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  glm::vec3 position = depthbuffer[index].position;
	  glm::vec3 color = depthbuffer[index].color;
	  glm::vec3 N = depthbuffer[index].normal;

	  glm::vec3 L = glm::normalize(lightPos - position);

	  float diffuse = max(glm::dot(N,L), 0.0);

	  //depthbuffer[index].color = glm::vec3(position.z > -100);
	  //depthbuffer[index].color = (N + 1.0f)/2.0f;
	  //depthbuffer[index].color = glm::vec3((position.z + 0.9579)*0.3158 );
	  //depthbuffer[index].color = color;
	  depthbuffer[index].color = glm::clamp(color*diffuse, 0.0, 1.0);
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

int numIterations = 0;
float vsTime = 0.0f;
float paTime = 0.0f;
float rTime = 0.0f;
float fsTime = 0.0f;
float cpTime = 0.0f;

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float cameraAngleA, float cameraAngleB){
#define SHAPE_TYPE 2
#if SHAPE_TYPE == 0
  float centerY = 0.05f;
  float dist = 0.1f;
#elif SHAPE_TYPE == 1
  float centerY = 0.1f;
  float dist = 0.3f;
#else
	// Cow: x=-0.4848..0.4848, y=0..0.5938, z=-0.1579..0.1579
  float centerY = 0.2969f;
  float dist = 0.8f;
#endif
  float cameraX = dist * sin(cameraAngleA) * cos(cameraAngleB);
  float cameraY = centerY + dist * sin(cameraAngleB);
  float cameraZ = dist * cos(cameraAngleA) * cos(cameraAngleB);
  float upX = -sin(cameraAngleA) * sin(cameraAngleB);
  float upY = cos(cameraAngleB);
  float upZ = -cos(cameraAngleA) * sin(cameraAngleB);

  // Projection matrix : 45deg Field of View, 1:1 ratio, display range : 0.01 unit <-> 100 units
  glm::mat4 Projection = glm::perspective(90.0f, 1.0f / 1.0f, 0.01f, 100.0f);
  // Camera matrix
  glm::mat4 View       = glm::lookAt(
    glm::vec3(cameraX,cameraY,cameraZ), // Camera position in World Space
    glm::vec3(0,centerY,0),                 // Looks towards
    glm::vec3(upX,upY,upZ)              // Up
  );
  // Model matrix : an identity matrix (model will be at the origin)
  glm::mat4 Model      = glm::mat4(1.0f);
  // Our ModelViewProjection : multiplication of our 3 matrices
  glm::mat4 MVP        = Projection * View * Model;
  glm::mat4 MV         = View * Model;

  glm::vec3 lightPos(1.0);
  lightPos = glm::vec3((MV * glm::vec4(lightPos, 1.0)));


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
  float* device_vbo2 = NULL;
  cudaMalloc((void**)&device_vbo, vbosize*sizeof(float));
  cudaMalloc((void**)&device_vbo2, vbosize*sizeof(float));
  cudaMemcpy( device_vbo, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------
  clock_t t0 = clock();
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, MVP, MV, device_vbo2);

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  clock_t t1 = clock();
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, device_vbo2);

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  clock_t t2 = clock();
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  clock_t t3 = clock();
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, lightPos);

  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  clock_t t4 = clock();
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();
  clock_t t5 = clock();

  float temp;
  numIterations ++;
  temp = ((float)(t1 - t0))/CLOCKS_PER_SEC;
  vsTime += (temp - vsTime) / (float) numIterations;
  temp = ((float)(t2 - t1))/CLOCKS_PER_SEC;
  paTime += (temp - paTime) / (float) numIterations;
  temp = ((float)(t3 - t2))/CLOCKS_PER_SEC;
  rTime += (temp - rTime) / (float) numIterations;
  temp = ((float)(t4 - t3))/CLOCKS_PER_SEC;
  fsTime += (temp - fsTime) / (float) numIterations;
  temp = ((float)(t5 - t4))/CLOCKS_PER_SEC;
  cpTime += (temp - cpTime) / (float) numIterations;

  std::cout << vsTime << "," << paTime << "," << rTime << "," << fsTime << "," << cpTime << std::endl;

  kernelCleanup();
  cudaFree( device_vbo2 );

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

