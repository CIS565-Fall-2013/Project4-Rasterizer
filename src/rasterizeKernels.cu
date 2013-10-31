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

bool POINTS=false;
enum coloringScheme {DIFFUSE_WHITE, PERVERTEXCOLOR, PERTRIANGLECOLOR, NORMALCOLORS, DEPTH};
int colScheme = DIFFUSE_WHITE;
int light=1;
int numBRDFS = 2;

int w=800; int h=800;
float fovy = 45;
float near = 1.0f;
#define far 10000.0f

glm::mat4 model = glm::mat4(glm::vec4( 1.0f, 0.0f, 0.0f, 0.0f),
							glm::vec4( 0.0f, 1.0f, 0.0f, 0.0f),
							glm::vec4( 0.0f, 0.0f, 1.0f, 0.0f),
							glm::vec4( 0.0f, 0.0f, 0.0f, 1.0f));
glm::vec3 eye = glm::vec3(0.0f,0.0f,6.0f);
glm::vec3 look = glm::vec3(0.0f,0.0f,0.0f);
glm::vec3 up = glm::vec3(0.0f,1.0f,0.0f);
glm::mat4 view =  glm::lookAt(eye,look,up);

glm::mat4 projection = glm::perspective(fovy, w * 1.0f/h,near,far);

glm::mat4 MVP = projection * view * model;



cam::cam()
{
	rad=2*glm::length(eye);
	theta=90.0;
	phi=0.0;
	pos = eye;
	idle = true;
	delPhi = 0.1f;
}
void cam::reset()
{
	rad=2*glm::length(eye);
	theta=90.0;
	phi=45.0;
	pos = eye;
}
void cam::setFrame()
{	
	pos.x=rad*sin(3.14*theta/180)*sin(3.14*phi/180);
	pos.z=rad*sin(3.14*theta/180)*cos(3.14*phi/180);
	pos.y=rad*cos(3.14*theta/180);
}


#if BACKFACECULLING
  struct checkNormal
  {
	  __host__ __device__ bool operator() (const triangle thisTriangle)
	  {
		  return (thisTriangle.n.z < 0);
	  }
  };
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
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
  
  if(x<resolution.x && y<resolution.y){

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
      
	  int px = resolution.x - x;
	  int py = resolution.y - y;
	  int pboIndex = px + (py * resolution.x);

      // Each thread writes one pixel location in the texture (textel)
      PBOpos[pboIndex].w = 0;
      PBOpos[pboIndex].x = color.x;     
      PBOpos[pboIndex].y = color.y;
      PBOpos[pboIndex].z = color.z;
  }
}

//TODO: Implement a vertex shader
__global__ void vertexShadeKernel(float* vbo, float *nbo, int vbosize, glm::mat4 modelView, glm::mat4 proj){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/vertexStride){
	  // create position and normal vector
	  glm::vec4 position = glm::vec4(vbo[vertexStride*index+0], vbo[vertexStride*index+1], 
									 vbo[vertexStride*index+2], vbo[vertexStride*index+3]);
	  glm::vec4 normal   = glm::vec4(nbo[vertexStride*index+0], nbo[vertexStride*index+1], 
									 nbo[vertexStride*index+2], nbo[vertexStride*index+3]);
#if DEBUG==0
	  // transform position
	  position = proj * modelView * position;
	  
	  // CHECK THIS!
	  normal   = (glm::transpose(glm::inverse(modelView)) * normal);
#else
	  position = modelView * position;
	  position = proj * position;

	  glm::mat4 normalMatrix = glm::inverse(modelView);
	  normalMatrix = glm::transpose(normalMatrix);

	  normal = normalMatrix * normal;

#endif
	  normal = normal;// /normal.w;
	  normal.w = 0;
	  normal = glm::normalize(normal);

	  // put back floats
	  vbo[vertexStride*index+0] = position.x;
	  vbo[vertexStride*index+1] = position.y;
	  vbo[vertexStride*index+2] = position.z;
	  vbo[vertexStride*index+3] = position.w;

	  nbo[vertexStride*index+0] = normal.x;
	  nbo[vertexStride*index+1] = normal.y;
	  nbo[vertexStride*index+2] = normal.z;
	  nbo[vertexStride*index+3] = normal.w;
	  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, float * nbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, int cScheme){
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

	  glm::vec3 n0 = glm::vec3(nbo[vertexStride* i0+0], nbo[vertexStride * i0+1], nbo[vertexStride * i0+2]);
	  glm::vec3 n1 = glm::vec3(nbo[vertexStride* i1+0], nbo[vertexStride * i1+1], nbo[vertexStride * i1+2]);
	  glm::vec3 n2 = glm::vec3(nbo[vertexStride* i2+0], nbo[vertexStride * i2+1], nbo[vertexStride * i2+2]);
	  
	  glm::vec3 c0, c1, c2;
	  if(cScheme == PERTRIANGLECOLOR || cScheme == PERVERTEXCOLOR)
	  {
	  // Get colors :: implementation: colors are given based on index of triangle, alternating between red, blue and green
	  int numCols = cbosize/colorStride;
	  int nextColor = index%numCols;
	  c0 = glm::vec3(cbo[colorStride * nextColor+0], cbo[colorStride * nextColor+1], cbo[colorStride * nextColor+2]);
	  nextColor=(nextColor + 1) %numCols;
	  c1 = glm::vec3(cbo[colorStride * nextColor+0], cbo[colorStride * nextColor+1], cbo[colorStride * nextColor+2]);
	  nextColor=(nextColor + 1) %numCols;
	  c2 = glm::vec3(cbo[colorStride * nextColor+0], cbo[colorStride * nextColor+1], cbo[colorStride * nextColor+2]);
	  }
	if(cScheme == DIFFUSE_WHITE)
	{
		  c0 = glm::vec3(1,1,1);
		  c2=c1=c0;
	}
	if(cScheme == PERTRIANGLECOLOR)
		  c2 = c1 = c0;


	  // Assemble primitive
	  thisTriangle.c0 = c0;	thisTriangle.c1 = c1;	thisTriangle.c2 = c2;
	  thisTriangle.p0 = p0; thisTriangle.p1 = p1;	thisTriangle.p2 = p2;
	  thisTriangle.n = (n0+n1+n2)/3.0f;

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
		f.position.z = depth;//getValueAtCoordinate(bary,thisTriangle.p0.z,thisTriangle.p1.z,thisTriangle.p2.z);
		// redundant to calculate x y z? No, because we need clip space coordinates for these lighting calculations

		f.normal = thisTriangle.n;

		f.locked = 1;

		//lock(&depthbuffer[bufferIndex].locked);
		if(depth > depthbuffer[bufferIndex].position.z && f.position.z <= 1 )
			depthbuffer[bufferIndex] = f;
		//unlock(&depthbuffer[bufferIndex].locked);
#endif

	}
}

#else

__device__ void serialRaster(triangle thisTriangle, glm::vec3 normal, fragment* depthbuffer, glm::vec2 resolution, int loX, int loY, int hiX, int hiY)
{
#if BACKFACEIGNORING
	if(thisTriangle.n.z < 0)
		return;
#endif

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
					f.position.z = depth;//getValueAtCoordinate(bary,thisTriangle.p0.z,thisTriangle.p1.z,thisTriangle.p2.z);
					// redundant to calculate x y z? No, because we need clip space coordinates for these lighting calculations

					f.normal = thisTriangle.n;

					f.locked = 1;

					//lock(&depthbuffer[bufferIndex].locked);
					if(depth > depthbuffer[bufferIndex].position.z && f.position.z <= 1 )
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

__global__ void pointRasterKernel(float* vbo, int vbosize, fragment* depthbuffer, glm::vec2 resolution)
{
	int index = (blockIdx.x * blockDim.x) + threadIdx.x;
	if(index<vbosize/vertexStride){
		glm::vec4 position = glm::vec4(vbo[vertexStride*index+0], vbo[vertexStride*index+1], vbo[vertexStride*index+2],0); // only care about projected components

		int x = glm::round(position.x);
		int y = glm::round(position.y);

		if( (x >= 0 && x < resolution.x) && ( y >= 0 && y < resolution.y))
		{
			fragment f;
			f.position = glm::vec3(position.x, position.y, position.z);
			f.normal = glm::vec3(0,0,1);
			f.color = glm::vec3(0.25,0.65,0.85);
			f.locked = 0;

			int bufferindex = x + y*resolution.x;
			depthbuffer[bufferindex] = f;
		}
	}
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, int cScheme, int lighting){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){

	// Read from buffer
	fragment thisFrag = depthbuffer[index];
	glm::vec3 lightPos;

	if(lighting==1)
	{
		// diffuse lighting
		lightPos = LIGHTPOS;
		glm::vec3 lightToPos = glm::normalize(lightPos - thisFrag.position);
		float diffuseTerm = glm::clamp(glm::dot(lightToPos,thisFrag.normal),0.0f,1.0f);

		thisFrag.color = diffuseTerm * thisFrag.color;
	}

	/*
	thisFrag.color.x = x * 1.0f/resolution.x;
	thisFrag.color.y = y * 1.0f/resolution.y;
	thisFrag.color.z = 1.0f;
	*/
	if(cScheme == NORMALCOLORS)
		thisFrag.color = glm::clamp(thisFrag.normal,0.0f,1.0f);
	else if(cScheme == DEPTH && thisFrag.position.z > -10000)
		thisFrag.color = glm::vec3(fabs(thisFrag.position.z)/(far));
	/*
	thisFrag.color.x = fabs(thisFrag.normal.x);
	thisFrag.color.y = fabs(thisFrag.normal.y);
	thisFrag.color.z = fabs(thisFrag.normal.z);
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, float * nbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  mouseCam.setFrame();

#if PERFANALYZE
  float vsTime = 0;
  float paTime = 0;
  float bfTime = 0;
  float rzTime = 0;
  float fsTime = 0;
#endif

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

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, vbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

#if PERFANALYZE
// Performance Analysis End
  cudaEvent_t start,stop;
  // Generate events
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Trigger event 'start'
  cudaEventRecord(start, 0);
#endif
  //------------------------------
  //vertex shader
  //------------------------------
  //vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, glm::mat4(1.0f));
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_nbo, vbosize, glm::lookAt(mouseCam.pos,look,up) *model, projection);
  
  //------------------------------
  //vertex shader
  //------------------------------
  perspectiveViewportTransform<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, resolution, glm::vec2(near,far));

  cudaDeviceSynchronize();

#if PERFANALYZE
  // Performance Analysis End
  cudaEventRecord(stop, 0); // Trigger Stop event
  cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)

  cudaEventElapsedTime(&vsTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
#endif

  if(!POINTS)
  {
	  //------------------------------
	  //primitive assembly
	  //------------------------------
	  int numTriangles = ibosize/3;
	  primitiveBlocks = ceil(((float)numTriangles)/((float)tileSize));
#if PERFANALYZE
	  // Trigger event 'start'
	  cudaEventRecord(start, 0);
#endif
	  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_nbo, vbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, colScheme);

	  cudaDeviceSynchronize();
#if PERFANALYZE
	  cudaEventRecord(stop, 0); // Trigger Stop event
	  cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)

	  cudaEventElapsedTime(&paTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
#endif

#if BACKFACECULLING
	  //------------------------------
	  // Stream Compact Back Face
	  //------------------------------
#if PERFANALYZE
	  // Trigger event 'start'
	  cudaEventRecord(start, 0);
#endif

	  thrust::device_ptr<triangle> thrust_primitives = thrust::device_pointer_cast(primitives);
	  triangle *  thrust_new_prims = thrust::remove_if(thrust_primitives, thrust_primitives+numTriangles, checkNormal()).get();
	  numTriangles = thrust_new_prims - primitives;
	  primitiveBlocks = ceil(((float)numTriangles)/((float)tileSize));

#if PERFANALYZE
	  cudaEventRecord(stop, 0); // Trigger Stop event
	  cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)

	  cudaEventElapsedTime(&bfTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
#endif

#if DEBUG
	  std::cout<<"thrust compaction: new number of triangles: "<<numTriangles<<std::endl;
#endif

#endif

#if PERFANALYZE
  	  // Trigger event 'start'
	  cudaEventRecord(start, 0);
#endif
	  //------------------------------
	  //rasterization
	  //------------------------------
	  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, ibosize/3, depthbuffer, resolution);
	  cudaDeviceSynchronize();

#if PERFANALYZE
  	  cudaEventRecord(stop, 0); // Trigger Stop event
	  cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)

	  cudaEventElapsedTime(&rzTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
#endif
  }
  else
  {

#if PERFANALYZE
    // Trigger event 'start'
	cudaEventRecord(start, 0);
#endif
	pointRasterKernel<<<primitiveBlocks, tileSize>>>(device_vbo, vbosize, depthbuffer, resolution);
	cudaDeviceSynchronize();
#if PERFANALYZE	
	cudaEventRecord(stop, 0); // Trigger Stop event
	cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)

	cudaEventElapsedTime(&rzTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
#endif
  }
  
  //------------------------------
  //fragment shader
  //------------------------------
#if PERFANALYZE
  // Trigger event 'start'
  cudaEventRecord(start, 0);
#endif
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution, colScheme, light);

  cudaDeviceSynchronize();
#if PERFANALYZE
  cudaEventRecord(stop, 0); // Trigger Stop event
  cudaEventSynchronize(stop); // Sync events (BLOCKS till last (stop in this case) has been recorded!)

  cudaEventElapsedTime(&fsTime, start, stop); // Calculate runtime, write to elapsedTime -- cudaEventElapsedTime returns value in milliseconds. Resolution ~0.5ms
#endif

  //------------------------------
  //write fragments to framebuffer
  //------------------------------
  render<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer, framebuffer);
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer);

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAError("Kernel failed!");
#if PERFANALYZE
  printf(" vs: %2.3fms  pa: %2.3fms  bf:  %2.3fms  rz  %2.3fms  fs %2.3fms\n",vsTime,paTime,bfTime,rzTime,fsTime);
#endif
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_nbo );
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

