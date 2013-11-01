// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include <thrust/device_ptr.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

//#define backfaceCulling
#define antialiasing 1

glm::vec3 *sFramebuffer;
glm::vec3* framebuffer;
float *depthBuffer;
int *dBufferLocked;
varying* interpVariables;
float* device_vbo;
float* device_vbo_eye;
float* device_nbo;
float* device_cbo;
int* device_ibo;
triangle* primitives;

struct primitive_is_culled{  
	__host__ __device__  bool operator()(const triangle tri)  
	{    
		return tri.toBeDiscard == 1;  
	}
};

void checkCUDAError(const char *msg, int line = -1)
{
    cudaError_t err = cudaGetLastError();
    if( cudaSuccess != err)
    {
        if( line >= 0 )
        {
            fprintf(stderr, "Line %d: ", line);
        }
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


__host__ __device__ void printMat4(glm::mat4 m){
    printf("%f, %f, %f, %f;\n", m[0][0], m[1][0], m[2][0], m[3][0]);
    printf("%f, %f, %f, %f;\n", m[0][1], m[1][1], m[2][1], m[3][1]);
    printf("%f, %f, %f, %f;\n", m[0][2], m[1][2], m[2][2], m[3][2]);
    printf("%f, %f, %f, %f;\n", m[0][3], m[1][3], m[2][3], m[3][3]);
}

__host__ __device__ glm::vec3 reflect(glm::vec3 I, glm::vec3 N)
{
	glm::vec3 R = I - 2.0f * N * glm::dot(I, N);
	return R;
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
__global__ void clearDepthBuffer(glm::vec2 resolution, varying* buffer, float *depthBuffer, int *dBufferLocked, varying frag, float zFar){
    int x = (blockIdx.x * blockDim.x) + threadIdx.x;
    int y = (blockIdx.y * blockDim.y) + threadIdx.y;
    int index = x + (y * resolution.x);
    if(x<=resolution.x && y<=resolution.y){
      varying f = frag;
	  f.normal = glm::vec3(0.0f);
      f.position.x = x;
      f.position.y = y;
      buffer[index] = f;
//	  unsigned int *zFarPtr = (unsigned int *)&zFar;
//	  depthBuffer[index] = FloatFlip((unsigned int&)*zFarPtr);
	  depthBuffer[index] = zFar;
	  dBufferLocked[index] = 0;
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
	  /*printf("worldCoords: \n");
	  printVec4(vertex);*/
	  vertex = view*vertex;
	  /*printf("eyeCoords: \n");
	  printVec4(vertex);*/
	  vbo_eye[3*index]   = vertex.x;
	  vbo_eye[3*index+1] = vertex.y;
	  vbo_eye[3*index+2] = vertex.z;
	  	 
	  // transform normal to eye space
	  normal = glm::transpose(glm::inverse(view))*normal;
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

	  /*glm::vec3 screenVet(vbo[3*index], vbo[3*index+1], vbo[3*index+2]);
	  printf("windowCoords: \n");
	  printVec3(screenVet);*/
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo, float *vbo_eye, int vbosize, float *nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize, triangle* primitives, glm::vec2 resolution, float zNear, float zFar){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  primitives[index].p0.x = vbo[3*ibo[3*index]];     primitives[index].p0.y = vbo[3*ibo[3*index]+1];     primitives[index].p0.z = vbo[3*ibo[3*index]+2];
	  primitives[index].p1.x = vbo[3*ibo[3*index+1]];   primitives[index].p1.y = vbo[3*ibo[3*index+1]+1];   primitives[index].p1.z = vbo[3*ibo[3*index+1]+2];
	  primitives[index].p2.x = vbo[3*ibo[3*index+2]];   primitives[index].p2.y = vbo[3*ibo[3*index+2]+1];   primitives[index].p2.z = vbo[3*ibo[3*index+2]+2];

	  primitives[index].eyeCoords0.x = vbo_eye[3*ibo[3*index]];     primitives[index].eyeCoords0.y = vbo_eye[3*ibo[3*index]+1];      primitives[index].eyeCoords0.z = vbo_eye[3*ibo[3*index]+2];
	  primitives[index].eyeCoords1.x = vbo_eye[3*ibo[3*index+1]];   primitives[index].eyeCoords1.y = vbo_eye[3*ibo[3*index+1]+1];    primitives[index].eyeCoords1.z = vbo_eye[3*ibo[3*index+1]+2];
	  primitives[index].eyeCoords2.x = vbo_eye[3*ibo[3*index+2]];   primitives[index].eyeCoords2.y = vbo_eye[3*ibo[3*index+2]+1];    primitives[index].eyeCoords2.z = vbo_eye[3*ibo[3*index+2]+2];

	  primitives[index].eyeNormal0.x = nbo[3*ibo[3*index]];      primitives[index].eyeNormal0.y = nbo[3*ibo[3*index]+1];     primitives[index].eyeNormal0.z = nbo[3*ibo[3*index]+2];
	  primitives[index].eyeNormal1.x = nbo[3*ibo[3*index+1]];    primitives[index].eyeNormal1.y = nbo[3*ibo[3*index+1]+1];   primitives[index].eyeNormal1.z = nbo[3*ibo[3*index+1]+2];
	  primitives[index].eyeNormal2.x = nbo[3*ibo[3*index+2]];    primitives[index].eyeNormal2.y = nbo[3*ibo[3*index+2]+1];   primitives[index].eyeNormal2.z = nbo[3*ibo[3*index+2]+2];

	  primitives[index].c0.x = cbo[0];		   primitives[index].c0.y = cbo[1];         primitives[index].c0.z = cbo[2];  
	  primitives[index].c1.x = cbo[3];         primitives[index].c1.y = cbo[4];		    primitives[index].c1.z = cbo[5];
	  primitives[index].c2.x = cbo[6];         primitives[index].c2.y = cbo[7];         primitives[index].c2.z = cbo[8];

	  primitives[index].toBeDiscard = 0;

#if defined(backfaceCulling)
	  if(calculateSignedArea(primitives[index]) < 1e-6) primitives[index].toBeDiscard = 1; // back facing triangles
	  else    // triangles totally outside of screen
	  {
		  glm::vec3 triMin, triMax;
		  getAABBForTriangle(primitives[index], triMin, triMax);
		  if(triMin.x > resolution.x || triMin.y > resolution.y || triMin.z > zFar || 
			 triMax.x < 0            || triMax.y < 0            || triMax.z < zNear) 
				 primitives[index].toBeDiscard = 1;
	  }
	  
#endif
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, varying* interpVariables, float *depthBuffer, int *dBufferLocked, glm::vec2 resolution, float zNear, float zFar){
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
			 triMax.x < 0            || triMax.y < 0            || triMax.z < zNear) 
		  {
				 //printf("triangle %d is outside!\n", index);
				 return; 
		  }
		  else 
		  {
			  /*printf("bounding box min: \n");
			  printVec3(triMin);
			  printf("bounding box max: \n");
			  printVec3(triMax);*/
			  glm::vec2 pixelCoords;
			  float depth;
			  glm::vec3 barycentricCoords;
			  int pixelIndex;
			  for(int j = max(int(triMin.y), 0); j < min(int(triMax.y+1), int(resolution.y)); ++j)
			  {
				  glm::vec2 Q0(triMin.x, float(j+0.5));
				  glm::vec2 Q1(triMax.x, float(j+0.5));
				  glm::vec2 u = Q1 - Q0;
				  float s;
				  float t;
				  float minS = 1.0f, maxS = 0.0f;
				  glm::vec2 v0((thisTriangle.p1 - thisTriangle.p0).x, (thisTriangle.p1 - thisTriangle.p0).y);
				  glm::vec2 v1((thisTriangle.p2 - thisTriangle.p1).x, (thisTriangle.p2 - thisTriangle.p1).y);
				  glm::vec2 v2((thisTriangle.p0 - thisTriangle.p2).x, (thisTriangle.p0 - thisTriangle.p2).y);

				  glm::vec2 w;
				  if(abs(u.x*v0.y - u.y*v0.x) > 1e-6) // not parallel
				  {
					  w = Q0 - glm::vec2(thisTriangle.p0.x, thisTriangle.p0.y);
					  s = (v0.y*w.x - v0.x*w.y) / (v0.x*u.y - v0.y*u.x);
					  t = (u.x*w.y  - u.y*w.x ) / (u.x*v0.y - u.y*v0.x);
					  if(s > -1e-6 && s < 1+1e-6 && t > -1e-6 && t < 1+1e-6)
					  {
						  minS = fminf(s, minS);
						  maxS = fmaxf(s, maxS);
					  }
				  }
				  if(abs(u.x*v1.y - u.y*v1.x) > 1e-6) // not parallel
				  {
					  w = Q0 - glm::vec2(thisTriangle.p1.x, thisTriangle.p1.y);
					  s = (v1.y*w.x - v1.x*w.y) / (v1.x*u.y - v1.y*u.x);
					  t = (u.x*w.y  - u.y*w.x ) / (u.x*v1.y - u.y*v1.x);
					  if(s > -1e-6 && s < 1+1e-6 && t > -1e-6 && t < 1+1e-6)
					  {
						  minS = fminf(s, minS);
						  maxS = fmaxf(s, maxS);
					  }
				  }
				  if(abs(u.x*v2.y - u.y*v2.x) > 1e-6) // not parallel
				  {
					  w = Q0 - glm::vec2(thisTriangle.p2.x, thisTriangle.p2.y);
					  s = (v2.y*w.x - v2.x*w.y) / (v2.x*u.y - v2.y*u.x);
					  t = (u.x*w.y  - u.y*w.x ) / (u.x*v2.y - u.y*v2.x);
					  if(s > -1e-6 && s < 1+1e-6 && t > -1e-6 && t < 1+1e-6)
					  {
						  minS = fminf(s, minS);
						  maxS = fmaxf(s, maxS);
					  }
				  }
				  
				  for(int i = max(int(triMin.x + minS * u.x), 0); i < min(int(triMin.x + maxS * u.x + 1), int(resolution.x)); ++i)
				  {
					  pixelCoords = glm::vec2(float(i+0.5), float(j+0.5));
					  barycentricCoords = calculateBarycentricCoordinate(thisTriangle, pixelCoords);
					  depth = barycentricCoords.x * thisTriangle.p0.z + barycentricCoords.y * thisTriangle.p1.z + barycentricCoords.z * thisTriangle.p2.z;	
					  pixelIndex = resolution.x - 1 - i + ((resolution.y  - 1 - j) * resolution.x);

					  if(isBarycentricCoordInBounds(barycentricCoords) && depth > zNear && depth < zFar)
					  {
						  bool wait = true;
						  //do{} while(atomicCAS(&dBufferLocked[pixelIndex], 0, 1)); 
						  while(wait)
						  {
							  if(0 == atomicExch(&dBufferLocked[pixelIndex], 1))
							  {
								   
								  if(depth < depthBuffer[pixelIndex]) 
								  {
									  depthBuffer[pixelIndex] = depth;
					  				  interpVariables[pixelIndex].position = barycentricCoords.x * thisTriangle.eyeCoords0 + barycentricCoords.y * thisTriangle.eyeCoords1 + barycentricCoords.z * thisTriangle.eyeCoords2;						  
					  				  interpVariables[pixelIndex].normal   = barycentricCoords.x * thisTriangle.eyeNormal0 + barycentricCoords.y * thisTriangle.eyeNormal1 + barycentricCoords.z * thisTriangle.eyeNormal2; 
									  interpVariables[pixelIndex].color    = barycentricCoords.x * thisTriangle.c0         + barycentricCoords.y * thisTriangle.c1         + barycentricCoords.z * thisTriangle.c2; 
								  }
								  dBufferLocked[pixelIndex] = 0;
								  wait = false;
							  }
						  }
					  }	
				  }
			  }
		  }
	  }
   }
}


__global__ void downSamplingKernel(glm::vec2 resolution, glm::vec3* sFramebuffer, glm::vec3* framebuffer){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
	  int superSampledIndex;
	  int baseIndex = antialiasing*antialiasing*index - antialiasing*(index%((int)resolution.x))*(antialiasing-1);
	  float weight = 1 / (float)(antialiasing*antialiasing);
	  glm::vec2 sResolution = (float)antialiasing*resolution;
	 // framebuffer[index] = sFramebuffer[baseIndex];
	  for(int j = 0; j < antialiasing; ++j)
	  {
		  for(int i = 0; i < antialiasing; ++i)
		  {
			  superSampledIndex = baseIndex + j*sResolution.x + i;
			  framebuffer[index] += weight*sFramebuffer[superSampledIndex];
		  }
	  }

  }

}
//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(varying* interpVariables, glm::vec2 resolution, glm::vec3 lightPosition, glm::vec3* framebuffer){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  
  if(x<=resolution.x && y<=resolution.y){
	  float specular = 400.0;
	  float ka = 0.1;
	  float kd = 0.7;
	  float ks = 0.2;
	  varying inVariables = interpVariables[index];

	  glm::vec3 lightVector = glm::normalize(lightPosition - inVariables.position);  // watch out for division by zero
	  glm::vec3 normal = glm::normalize(inVariables.normal); // watch out for division by zero
	  float diffuseTerm = glm::clamp(glm::dot(normal, lightVector), 0.0f, 1.0f);

	  glm::vec3 R = glm::normalize(reflect(-lightVector, normal)); // watch out for division by zero
	  glm::vec3 V = glm::normalize(- inVariables.position); // watch out for division by zero
      float specularTerm = pow( fmaxf(glm::dot(R, V), 0.0f), specular );

	  framebuffer[index] = ka*inVariables.color + glm::vec3(1.0f) * (kd*inVariables.color*diffuseTerm + ks*specularTerm);
	  // framebuffer[index] = inVariables.normal;
	  // framebuffer[index] = glm::vec3(1.0f) * inVariables;
  }
}

//Writes fragment colors to the framebuffer
/*
__global__ void render(glm::vec2 resolution, varying* interpVariables, glm::vec3* framebuffer){// write true color to framebuffer

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[index] = interpVariables[index].color;
  }
}*/

// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, glm::mat4 projection, glm::mat4 view, float zNear, float zFar, glm::vec3 lightPosition, float* vbo, int vbosize, float *nbo, int nbosize, float* cbo, int cbosize, int* ibo, int ibosize){

  glm::vec2 sResolution = (float)antialiasing * resolution;
  // set up crucial magic
  int tileSize = 8;
  dim3 threadsPerBlock(tileSize, tileSize);
  dim3 fullBlocksPerGrid((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));

  //set up framebuffer
  framebuffer = NULL;
  cudaMalloc((void**)&framebuffer, (int)resolution.x*(int)resolution.y*sizeof(glm::vec3)); // frame buffer store colors

  //set up supersampling framebuffer
  sFramebuffer = NULL;
  cudaMalloc((void**)&sFramebuffer, (int)sResolution.x*(int)sResolution.y*sizeof(glm::vec3)); 

  //set up depth buffer
  depthBuffer = NULL;
  cudaMalloc((void**)&depthBuffer, (int)sResolution.x*(int)sResolution.y*sizeof(float));

  dBufferLocked = NULL;
  cudaMalloc((void**)&dBufferLocked, (int)sResolution.x*(int)sResolution.y*sizeof(int));

  //set up interpVariables
  interpVariables = NULL;
  cudaMalloc((void**)&interpVariables, (int)sResolution.x*(int)sResolution.y*sizeof(varying)); // interpolation result per pixel by rasterizer

  //kernel launches to black out accumulated/unaccumlated pixel buffers and clear our scattering states
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, framebuffer, glm::vec3(0,0,0)); // launch kernel for every pixel
  
  fullBlocksPerGrid = dim3((int)ceil(float(sResolution.x)/float(tileSize)), (int)ceil(float(sResolution.y)/float(tileSize)));
  clearImage<<<fullBlocksPerGrid, threadsPerBlock>>>(sResolution, sFramebuffer, glm::vec3(0,0,0)); // launch kernel for every pixel
  checkCUDAErrorWithLine("Kernel failed!");

  //fullBlocksPerGrid = dim3((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
  varying frag;
  frag.color = glm::vec3(0,0,0);
  frag.normal = glm::vec3(1,0,0);
  frag.position = glm::vec3(0,0,-10000);
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(sResolution, interpVariables, depthBuffer, dBufferLocked, frag, zFar); // launch kernel for every pixel
  checkCUDAErrorWithLine("Kernel failed!");
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

  device_vbo_eye = NULL;
  cudaMalloc((void**)&device_vbo_eye, vbosize*sizeof(float)); 

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float)); 
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float)); 
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize)); // launch for every vertex
  //------------------------------
  //vertex shader
  //------------------------------
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(sResolution, projection, view, zNear, zFar, device_vbo, device_vbo_eye, vbosize, device_nbo, nbosize);
  checkCUDAErrorWithLine("Kernel failed!");
  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  int triCount = ibosize/3;
  printf("triangle count before culling: %d\n", triCount);
  primitiveBlocks = ceil(((float)triCount)/((float)tileSize)); // launch for every primitive
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_eye, vbosize, device_nbo, nbosize, device_cbo, cbosize, device_ibo, ibosize, primitives, sResolution, zNear, zFar);
  checkCUDAErrorWithLine("Kernel failed!");
  cudaDeviceSynchronize();
#if defined(backfaceCulling)// stream compaction to discard culled triangles
  thrust::device_ptr<triangle> primitive_first = thrust::device_pointer_cast(primitives);
  thrust::device_ptr<triangle> primitive_last  = thrust::remove_if(primitive_first, primitive_first + ibosize/3, primitive_is_culled());
  triCount = thrust::distance(primitive_first, primitive_last);	 
  printf("triangle count after culling: %d\n", triCount);
  checkCUDAErrorWithLine("Kernel failed!");
  cudaDeviceSynchronize();
#endif
  //------------------------------
  //rasterization
  //------------------------------
  primitiveBlocks = ceil(((float)triCount)/((float)tileSize)); 
  rasterizationKernel<<<primitiveBlocks, tileSize>>>(primitives, triCount, interpVariables, depthBuffer, dBufferLocked, sResolution, zNear, zFar); // launch for every primitive
  checkCUDAErrorWithLine("Kernel failed!");
  cudaDeviceSynchronize();
  checkCUDAErrorWithLine("Kernel failed!");
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec4 lightEyeSpace = view * glm::vec4(lightPosition, 1.0f);
  lightPosition = glm::vec3(lightEyeSpace.x, lightEyeSpace.y, lightEyeSpace.z);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(interpVariables, sResolution, lightPosition, sFramebuffer); // launch for every pixel
  checkCUDAErrorWithLine("Kernel failed!");
  cudaDeviceSynchronize();
  //------------------------------
  //write fragments to framebuffer after downsampling
  //------------------------------
  tileSize = 8;
  fullBlocksPerGrid = dim3((int)ceil(float(resolution.x)/float(tileSize)), (int)ceil(float(resolution.y)/float(tileSize)));
  downSamplingKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, sFramebuffer, framebuffer); // launch for every pixel
  sendImageToPBO<<<fullBlocksPerGrid, threadsPerBlock>>>(PBOpos, resolution, framebuffer); // launch for every pixel

  cudaDeviceSynchronize();

  kernelCleanup();

  checkCUDAErrorWithLine("Kernel failed!");
}

void kernelCleanup(){
  cudaFree( primitives );
  cudaFree( device_vbo );
  cudaFree( device_vbo_eye );
  cudaFree( device_cbo );
  cudaFree( device_nbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( sFramebuffer );
  cudaFree( depthBuffer );
  cudaFree( dBufferLocked );
  cudaFree( interpVariables );
}

