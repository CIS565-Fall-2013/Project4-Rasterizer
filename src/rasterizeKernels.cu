// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include "glm/gtc/matrix_transform.hpp"

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
	  f.position.z = -100000;
	  f.set = false;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, cudaMat4 MVP, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  
	  int vInd = index*3;
	  
	  //apply mvp transform to go to clip space and write back to VBO
	  glm::vec4 point(vbo[vInd],vbo[vInd+1], vbo[vInd+2], 1.0f);
	  point=multiplyMV_4(MVP, point);
	  
	  //perspective division to NDC and write to memory
	  point.x /= point.w;
	  point.y /= point.w;
	  point.z /= point.w;

	  //transfrom to screen coord
	  point.x = (point.x+1)*(resolution.x/2.0f);
	  point.y = (-point.y+1)*(resolution.y/2.0f);		//flip y coordinate
	  point.z = (zFar - zNear)/2.0f*point.z + (zFar + zNear)/2.0f;

	  //write to memory
	  vbo[vInd]=point.x;
	  vbo[vInd+1]=point.y;
	  vbo[vInd+2]=point.z;
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){

	  triangle tri=primitives[index];
	  
	  //get the vertices and colors of the triangle
	  int iInd = index*3;
	  int vInd = ibo[iInd]*3;

	  tri.p0 = glm::vec3(vbo[vInd], vbo[vInd+1], vbo[vInd+2]);
	  tri.c0 = glm::vec3(cbo[0], cbo[1], cbo[2]);

	  //get 2nd vertex
	  iInd++;
	  vInd = ibo[iInd]*3;
	  tri.p1 = glm::vec3(vbo[vInd], vbo[vInd+1], vbo[vInd+2]);
	  tri.c1 = glm::vec3(cbo[3], cbo[4], cbo[5]);

	  iInd++;
	  vInd = ibo[iInd]*3;
	  tri.p2 = glm::vec3(vbo[vInd], vbo[vInd+1], vbo[vInd+2]);
	  tri.c2 = glm::vec3(cbo[6], cbo[7], cbo[8]);

	  tri;

	  //write to memory
	  primitives[index]=tri;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  //need atomics to work....
	
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	  
	  //rasterize with barycentric method
	  triangle tri = primitives[index];

	  //find bounding box around triangle
	  glm::vec3 low, high;
	  getAABBForTriangle(tri, low, high);

	  //for each pixel in BB range, test if pixel is inside triangle

	  for (int i = clamp(low.x, 0.0f, resolution.x-1); i<=ceil(high.x) && i<resolution.x; ++i){
		  for(int j = clamp(low.y, 0.0f, resolution.y-1); j<=ceil(high.y) && j<resolution.y; ++j){
			  
			  int fragIndex = i*resolution.y + j;
			  fragment frag = depthbuffer[fragIndex];
			  
			  //convert pixel to barycentric coord
			  glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, glm::vec2(i, j));
			  
			  //check if within the triangle
			  if(isBarycentricCoordInBounds(baryCoord)){
				  float z = getZAtCoordinate(baryCoord, tri);
				  
				  //test depth
				  if(z > frag.position.z){
					  
					  glm::vec3 absCoord (abs(baryCoord.x), abs(baryCoord.y), abs(baryCoord.z));

					  frag.position = glm::vec3(i,j,z);
					  frag.normal = glm::normalize(glm::cross((tri.p0-tri.p1), (tri.p0-tri.p2)));
					  float tx = abs(frag.normal.x);
					  float ty = abs(frag.normal.y);
					  float tz = abs(frag.normal.z);
					  frag.color = glm::vec3(tx, ty, tz);
					  //frag.color = absCoord.x*tri.c0 + absCoord.y*tri.c1 + absCoord.z*tri.c2;
					  //frag.set = true;
					  depthbuffer[fragIndex] = frag;
				  }
			  }
		  }
	  }
  }
}

//per fragment rasterization
__global__ void rasterizationKernelFrag(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){

	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = x + (y*resolution.x);

	if(x <= resolution.x && y<=resolution.y){
		
		for(int i = 0; i < primitivesCount; ++i){
			
			triangle tri = primitives[i];
			fragment frag = depthbuffer[index];

			//find bounding box around triangle
			glm::vec3 low, high;
			getAABBForTriangle(tri, low, high);
			
			//throw out this fragment if ouside bounding box
			if(x<low.x || x>high.x || y<low.y || y>high.y)
				continue;

			//convert pixel to barycentric coord
			glm::vec3 baryCoord = calculateBarycentricCoordinate(tri, glm::vec2(x, y));

			//check if within the triangle
			if(isBarycentricCoordInBounds(baryCoord)){
				float z = getZAtCoordinate(baryCoord, tri);

				if(frag.set && frag.position.z<z)
					continue;

				frag.position = glm::vec3(x,y,z);
				frag.normal = glm::normalize(glm::cross((tri.p0-tri.p1), (tri.p0-tri.p2)));
				frag.color = glm::normalize(glm::vec3(abs(z/1000.0f)));
				//frag.color = frag.normal;
				frag.set = true;
			}
			depthbuffer[index] = frag;
		}
	
	}
}


//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	  //diffuse shading

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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, camera& cam){

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
  frag.position = glm::vec3(0,0,-100000);
  frag.set = false;
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

  //build the MVP matrix
  glm::mat4 model(1.0f);		//temp identity model view matrix for now
  glm::mat4 view = glm::lookAt(cam.eye, cam.center, cam.up);
  glm::mat4 projection = glm::perspective(cam.fov, 1.0f, cam.zNear, cam.zFar);
  glm::mat4 mvp = projection*view*model;

  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, utilityCore::glmMat4ToCudaMat4(mvp), resolution, cam.zNear, cam.zFar);
  checkCUDAErrorWithLine("vertex shade failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives);
  checkCUDAErrorWithLine("primitive assembly failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
  //rasterizationKernelFrag<<<fullBlocksPerGrid, threadsPerBlock>>>(primitives, ibosize/3, depthbuffer, resolution);
  checkCUDAErrorWithLine("rasterize kernel failed!");

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution);
  checkCUDAErrorWithLine("fragment failed!");

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

