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

int w=800; int h=800;
float fovy = 45;
float near = 0.1;
float far = 10000;

glm::mat4 model = glm::mat4(glm::vec4(-1.0f, 0.0f, 0.0f, 0.0f),
							glm::vec4( 0.0f,-1.0f, 0.0f, 0.0f),
							glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f),
							glm::vec4( 0.0f ,0.0f, 0.0f, 1.0f));
glm::vec3 eye = glm::vec3(0.0f,0.0f,9.0f);
glm::vec3 look = glm::vec3(0.0f,0.0f,0.0f);
glm::vec3 up = glm::vec3(0.0f,1.0f,0.0f);
glm::mat4 view =  glm::lookAt(eye,look,up);

glm::mat4 projection = glm::perspective(fovy, w * 1.0f/h,near,far);

glm::mat4 MVP = projection * view * model;

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
	  f.locked = 0;
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
__global__ void vertexShadeKernel(float* vbo, int vbosize, glm::mat4 modelViewProjection){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/vertexStride){
	  // create position vector
	  glm::vec4 position = glm::vec4(vbo[vertexStride*index+0], vbo[vertexStride*index+1], 
									 vbo[vertexStride*index+2], vbo[vertexStride*index+3]);
	  // transform position
	  position = modelViewProjection * position;

	  // put back floats
	  vbo[vertexStride*index+0] = position.x;
	  vbo[vertexStride*index+1] = position.y;
	  vbo[vertexStride*index+2] = position.z;
	  vbo[vertexStride*index+3] = position.w;

  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  triangle thisTriangle = primitives[index];
	  
	  // Get indices
	  int i0, i1, i2;
	  i0 = ibo[indexStride * index+0];
	  i1 = ibo[indexStride * index+1];
	  i2 = ibo[indexStride * index+2];
	  
	  // Get positions
	  glm::vec4 p0 = glm::vec4(vbo[vertexStride* i0+0], vbo[vertexStride * i0+1],
								vbo[vertexStride * i0+2], vbo[vertexStride * i0+3]);
	  glm::vec4 p1 = glm::vec4(vbo[vertexStride * i1+0], vbo[vertexStride * i1+1], 
								vbo[vertexStride * i1+2], vbo[vertexStride * i1+3]);
	  glm::vec4 p2 = glm::vec4(vbo[vertexStride * i2+0], vbo[vertexStride *i2 +1],
								vbo[vertexStride * i2+2], vbo[vertexStride * i2+3]);

	  // Get colors :: implementation: colors are given based on index of triangle, alternating between red, blue and green
	  int nextColor = index%cbosize;
	  glm::vec3 c0 = glm::vec3(cbo[colorStride * nextColor+0], cbo[colorStride * nextColor+1], cbo[colorStride * nextColor+2]);
	  nextColor=(nextColor + 1) %3;
	  glm::vec3 c1 = glm::vec3(cbo[colorStride * nextColor+0], cbo[colorStride * nextColor+1], cbo[colorStride * nextColor+2]);
	  nextColor=(nextColor + 1) %3;
	  glm::vec3 c2 = glm::vec3(cbo[colorStride * nextColor+0], cbo[colorStride * nextColor+1], cbo[colorStride * nextColor+2]);
	  nextColor=(nextColor + 1) %3;

	  // Assemble primitive
	  thisTriangle.c0 = c0;	thisTriangle.c1 = c1;	thisTriangle.c2 = c2;
	  thisTriangle.p0 = p0; thisTriangle.p1 = p1;	thisTriangle.p2 = p2;

	  // Write back the primitive
	  primitives[index] = thisTriangle;
  }
}

__global__ void perspectiveViewportTransform(float* vbo, float vbosize, glm::vec2 resolution, glm::vec2 nf)
{
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/vertexStride){
	  // create position vector
	  glm::vec4 position = glm::vec4(vbo[vertexStride*index+0], vbo[vertexStride*index+1], 
									 vbo[vertexStride*index+2], vbo[vertexStride*index+3]);
	  // transform position
	  position.x /= position.w;
	  position.y /= position.w;
	  position.z /= position.w;
	  
	  position.x = resolution.x/2  * position.x + resolution.x/2;
	  position.y = resolution.y/2  * position.y + resolution.y/2;
	  position.z = (nf.y - nf.x)/2 * position.z + (nf.y + nf.x)/2;

	  // put back floats
	  vbo[vertexStride*index+0] = position.x;
	  vbo[vertexStride*index+1] = position.y;
	  vbo[vertexStride*index+2] = position.z;
	  vbo[vertexStride*index+3] = position.w;

  }
}

#if DYNAMICPARALLELISM
__global__ void dynamicParallelRaster(triangle thisTriangle, glm::vec3 normal, fragment* depthbuffer, glm::vec2 resolution, int loX, int loY, int hiX, int hiY)
{
	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	// Check if within resolution
	if( (x > 0 && x < resolution.x) && (y > 0 && y < resolution.y) )
	{
		glm::vec3 bary = calculateBarycentricCoordinate(thisTriangle,glm::vec2(x,y));
	// Check if within projected triangle
		if( isBarycentricCoordInBounds(bary))
		{
	// Interpolate Data
			float depth = getZAtCoordinate(bary, thisTriangle);
			int bufferIndex = x + y * resolution.x;
#if DEPTHPRECHECK

	// Check if z is lower than already existing z
	// Use Atomics!
	// Check whether depth is +ve or -ve!

			lock(&depthbuffer[bufferIndex].locked);

			if(depth < depthbuffer[bufferIndex].position.z)
			{
	// ONLY then write to buffer
				fragment f;
	// Get interpolated color
				f.color.x = getValueAtCoordinate(bary,thisTriangle.c0.x,thisTriangle.c1.x,thisTriangle.c2.x);
				f.color.y = getValueAtCoordinate(bary,thisTriangle.c0.y,thisTriangle.c1.y,thisTriangle.c2.y);
				f.color.z = getValueAtCoordinate(bary,thisTriangle.c0.z,thisTriangle.c1.z,thisTriangle.c2.z);

				f.position.x = getValueAtCoordinate(bary,thisTriangle.p0.x,thisTriangle.p1.x,thisTriangle.p2.x);
				f.position.y = getValueAtCoordinate(bary,thisTriangle.p0.y,thisTriangle.p1.y,thisTriangle.p2.y);
				f.position.z = getValueAtCoordinate(bary,thisTriangle.p0.z,thisTriangle.p1.z,thisTriangle.p2.z);
	// redundant to calculate x y z? No, because we need clip space coordinates for these lighting calculations

				f.normal = normal;

				depthbuffer[bufferIndex] = f;
			}

			unlock(&depthbuffer[bufferIndex].locked);
#else
			fragment f;
	// Get interpolated color
			f.color.x = getValueAtCoordinate(bary,thisTriangle.c0.x,thisTriangle.c1.x,thisTriangle.c2.x);
			f.color.y = getValueAtCoordinate(bary,thisTriangle.c0.y,thisTriangle.c1.y,thisTriangle.c2.y);
			f.color.z = getValueAtCoordinate(bary,thisTriangle.c0.z,thisTriangle.c1.z,thisTriangle.c2.z);

			f.position.x = getValueAtCoordinate(bary,thisTriangle.p0.x,thisTriangle.p1.x,thisTriangle.p2.x);
			f.position.y = getValueAtCoordinate(bary,thisTriangle.p0.y,thisTriangle.p1.y,thisTriangle.p2.y);
			f.position.z = getValueAtCoordinate(bary,thisTriangle.p0.z,thisTriangle.p1.z,thisTriangle.p2.z);
	// redundant to calculate x y z? No, because we need clip space coordinates for these lighting calculations

			f.normal = normal;

			lock(&depthbuffer[bufferIndex].locked);
			depthbuffer[bufferIndex] = f;
			unlock(&depthbuffer[bufferIndex].locked);
#endif

		}
	}
}

#else

__device__ void serialRaster(triangle thisTriangle, glm::vec3 normal, fragment* depthbuffer, glm::vec2 resolution, int loX, int loY, int hiX, int hiY)
{

	for(int y = loY; y <= hiY; y++)
	{

#if OPTIMIZE_RASTER
		bool inside = false;
		int crossed = 0;
#endif

		for(int x = loX; x <= hiX; x++)
		{
			// Check if within resolution
			if( (x >= 0 && x < resolution.x) && (y >= 0 && y < resolution.y) )
			{
				glm::vec3 bary = calculateBarycentricCoordinate(thisTriangle,glm::vec2(x,y));
			// Check if within projected triangle
				if( isBarycentricCoordInBounds(bary))
				{
#if OPTIMIZE_RASTER
			//Optimize raster step
					if(inside==false)
					{
						inside = true;
						crossed++;
					}
#endif
			// Interpolate Data
					float depth = getZAtCoordinate(bary, thisTriangle);
					int bufferIndex = x + y * resolution.x;
#if DEPTHPRECHECK

			// Check if z is lower than already existing z
			// Use Atomics!
			// Check whether depth is +ve or -ve!

					lock(&depthbuffer[bufferIndex].locked);

					if(depth > depthbuffer[bufferIndex].position.z)
					{
			// ONLY then write to buffer
						fragment f;
						// Get interpolated color
						f.color.x = getValueAtCoordinate(bary,thisTriangle.c0.x,thisTriangle.c1.x,thisTriangle.c2.x);
						f.color.y = getValueAtCoordinate(bary,thisTriangle.c0.y,thisTriangle.c1.y,thisTriangle.c2.y);
						f.color.z = getValueAtCoordinate(bary,thisTriangle.c0.z,thisTriangle.c1.z,thisTriangle.c2.z);

						//@DO : REMOVE : TESTING
						f.color = glm::vec3(1,0,0);

						f.position.x = getValueAtCoordinate(bary,thisTriangle.p0.x,thisTriangle.p1.x,thisTriangle.p2.x);
						f.position.y = getValueAtCoordinate(bary,thisTriangle.p0.y,thisTriangle.p1.y,thisTriangle.p2.y);
						f.position.z = getValueAtCoordinate(bary,thisTriangle.p0.z,thisTriangle.p1.z,thisTriangle.p2.z);
						// redundant to calculate x y z? No, because we need clip space coordinates for these lighting calculations

						f.normal = normal;

						depthbuffer[bufferIndex] = f;
					}

					unlock(&depthbuffer[bufferIndex].locked);
#else
					fragment f;
					// Get interpolated color
					f.color.x = getValueAtCoordinate(bary,thisTriangle.c0.x,thisTriangle.c1.x,thisTriangle.c2.x);
					f.color.y = getValueAtCoordinate(bary,thisTriangle.c0.y,thisTriangle.c1.y,thisTriangle.c2.y);
					f.color.z = getValueAtCoordinate(bary,thisTriangle.c0.z,thisTriangle.c1.z,thisTriangle.c2.z);

					f.position.x = getValueAtCoordinate(bary,thisTriangle.p0.x,thisTriangle.p1.x,thisTriangle.p2.x);
					f.position.y = getValueAtCoordinate(bary,thisTriangle.p0.y,thisTriangle.p1.y,thisTriangle.p2.y);
					f.position.z = getValueAtCoordinate(bary,thisTriangle.p0.z,thisTriangle.p1.z,thisTriangle.p2.z);
					// redundant to calculate x y z? No, because we need clip space coordinates for these lighting calculations

					f.normal = normal;

					f.locked = 1;

					//lock(&depthbuffer[bufferIndex].locked);
					if(depth > depthbuffer[bufferIndex].position.z)
						depthbuffer[bufferIndex] = f;
					//unlock(&depthbuffer[bufferIndex].locked);
#endif

				}
#if OPTIMIZE_RASTER
				else
				{
					if(inside == true)
					{
						inside = false;
						crossed++;
					}
				}
#endif
			}
#if OPTIMIZE_RASTER
			// Break out of inner loop if crossed triangle twice.			
			if(crossed == 2)
				break;
#endif
		}
	}


}

#endif

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
	// Get Current Triangle
	triangle thisTriangle = primitives[index];

	glm::vec3 normal;

	// Triangle in NDC or Viewport?
	// Lets assume viewport coordinates, so from 0-0 to resolution x-y
	
	// Get triangle's bounding box
	glm::vec3 minPoint; 
	glm::vec3 maxPoint;
	getAABBForTriangle(thisTriangle,minPoint,maxPoint,resolution);

	// Get bounds for pixels to be rasterized
	int loX = glm::round(minPoint.x);
	int loY = glm::round(minPoint.y);
	int hiX = glm::round(maxPoint.x);
	int hiY = glm::round(maxPoint.y);

#if DYNAMICPARALLELISM
	// USE CUDA LAUNCHING KERNELS THINGY!

	int totX = hiX - loX;
	int totY = hiY - loY;

	// set up crucial magic
	int tileSize = 8;
	dim3 threadsPerBlock(tileSize, tileSize);
	dim3 fullBlocksPerGrid((int)ceil(float(totX)/float(tileSize)), (int)ceil(float(totY)/float(tileSize)));

	dynamicParallelRaster<<<fullBlocksPerGrid,threadsPerBlock>>>(thisTriangle,normal,depthbuffer,resolution, loX, loY, hiX, hiY);

#else

	serialRaster(thisTriangle,normal,depthbuffer,resolution, loX, loY, hiX, hiY);

#endif
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	// Read from buffer
	fragment thisFrag = depthbuffer[index];

	/*
	// diffuse lighting
	glm::vec3 lightPos = LIGHTPOS;
	glm::vec3 lightToPos = glm::normalize(lightPos - thisFrag.position);
	float diffuseTerm = glm::clamp(glm::dot(lightToPos,thisFrag.normal),0.0f,1.0f);

	thisFrag.color = diffuseTerm * thisFrag.color;
	*/
	/*
	thisFrag.color.x = x * 1.0f/resolution.x;
	thisFrag.color.y = y * 1.0f/resolution.y;
	thisFrag.color.z = 1.0f;
	*/


	// Write to buffer
	depthbuffer[index] = thisFrag;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize){

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
  //vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, glm::mat4(1.0f));
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, MVP);

  //------------------------------
  //vertex shader
  //------------------------------
  perspectiveViewportTransform<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, resolution, glm::vec2(near,far));

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

