// CIS565 CUDA Rasterizer: A simple rasterization pipeline for Patrick Cozzi's CIS565: GPU Computing at the University of Pennsylvania
// Written by Yining Karl Li, Copyright (c) 2012 University of Pennsylvania

#include <stdio.h>
#include <cuda.h>
#include <cmath>
#include <thrust/random.h>
#include "rasterizeKernels.h"
#include "rasterizeTools.h"
#include <thrust/device_ptr.h>
#include <thrust/partition.h>

#if CUDA_VERSION >= 5000
    #include <helper_math.h>
#else
    #include <cutil_math.h>
#endif

glm::vec3* framebuffer;
fragment* depthbuffer;
float* device_vbo;
float* device_nbo;
float* device_cbo;
float* device_vbo_worldspace;
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
__global__ void vertexShadeKernel(float* vbo,float* vboWorldSpace, int vbosize,  float* nbo, int nbosize,cudaMat4 mMatrix, cudaMat4 mvpMatrix, cudaMat4 mInverseTransMatrix,glm::vec2 resolution,glm::vec2 zplanes){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<vbosize/3){
	  glm::vec4 v(vbo[3*index], vbo[3*index + 1], vbo[3*index+2],1.0f);

	  glm::vec4 transformedVec = multiplyMV(mvpMatrix,v);
	  transformedVec.x = transformedVec.x/transformedVec.w;
	  transformedVec.y = transformedVec.y/transformedVec.w;
	  transformedVec.z = transformedVec.z/transformedVec.w;

	  cudaMat4 viewport;
	  viewport.x = glm::vec4(0.5f*resolution.x,0.0f,0.0f,0.5f*resolution.x);
	  viewport.y = glm::vec4(0.0f,0.5f*resolution.y,0.0f,0.5f*resolution.y);
	  viewport.z = glm::vec4(0.0f,0.0f,0.5f,0.5f);
	  viewport.w = glm::vec4(0.0f,0.0f,0.0f,1.0f);
	  
	  transformedVec.w = 1;
	  transformedVec = multiplyMV(viewport,transformedVec);

	  vbo[3*index] = transformedVec.x;
	  vbo[3*index + 1] = transformedVec.y;
	  vbo[3*index + 2] = transformedVec.z;

	  transformedVec = multiplyMV(mMatrix,v);
	  vboWorldSpace[3*index] = transformedVec.x;
	  vboWorldSpace[3*index + 1] = transformedVec.y;
	  vboWorldSpace[3*index + 2] = transformedVec.z;

	  glm::vec4 n( nbo[3*index], nbo[3*index + 1], nbo[3*index+2],0.0f);
	  glm::vec4 transformedNormal = multiplyMV(mInverseTransMatrix,n);
	  nbo[3*index]   = transformedNormal.x;
	  nbo[3*index+1] = transformedNormal.y;
	  nbo[3*index+2] = transformedNormal.z;
  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo,float* vboWorldSpace, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,float *nbo, int nbosize, triangle* primitives,cudaMat4 viewMatrix){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  int primitivesCount = ibosize/3;
  if(index<primitivesCount){
	  int firstVertexIdx = ibo[3*index];
	  int secondVertexIdx = ibo[3*index+1];
	  int thirdVertexIdx = ibo[3*index+2];
	  triangle t;
	  t.p0 = glm::vec3(vbo[3*firstVertexIdx], vbo[3*firstVertexIdx + 1], vbo[3*firstVertexIdx+2]);
	  t.p1 = glm::vec3(vbo[3*secondVertexIdx], vbo[3*secondVertexIdx + 1], vbo[3*secondVertexIdx+2]);
	  t.p2 = glm::vec3(vbo[3*thirdVertexIdx], vbo[3*thirdVertexIdx + 1], vbo[3*thirdVertexIdx+2]);
	  t.c0 = glm::vec3(cbo[0],cbo[1],cbo[2]);
	  t.c1 = glm::vec3(cbo[3],cbo[4],cbo[5]);
	  t.c2 = glm::vec3(cbo[6],cbo[7],cbo[8]);
	  t.n0 = glm::vec3(nbo[3*firstVertexIdx], nbo[3*firstVertexIdx + 1], nbo[3*firstVertexIdx+2]);
	  t.n1 = glm::vec3(nbo[3*secondVertexIdx],nbo[3*secondVertexIdx + 1], nbo[3*secondVertexIdx+2]);
	  t.n2 = glm::vec3(nbo[3*thirdVertexIdx], nbo[3*thirdVertexIdx + 1], nbo[3*thirdVertexIdx+2]);
	  t.p0world = glm::vec3(vboWorldSpace[3*firstVertexIdx], vboWorldSpace[3*firstVertexIdx + 1], vboWorldSpace[3*firstVertexIdx+2]);
	  t.p1world = glm::vec3(vboWorldSpace[3*secondVertexIdx], vboWorldSpace[3*secondVertexIdx + 1], vboWorldSpace[3*secondVertexIdx+2]);
	  t.p2world = glm::vec3(vboWorldSpace[3*thirdVertexIdx], vboWorldSpace[3*thirdVertexIdx + 1], vboWorldSpace[3*thirdVertexIdx+2]);
	  
	  glm::vec4 p0view = multiplyMV(viewMatrix,glm::vec4(t.p0world,1.0f));
	  glm::vec4 p1view = multiplyMV(viewMatrix,glm::vec4(t.p1world,1.0f));
	  glm::vec4 p2view = multiplyMV(viewMatrix,glm::vec4(t.p2world,1.0f));

	  glm::vec4 a = p1view - p0view;
	  glm::vec4 b = p2view - p0view;

	  glm::vec3 n = glm::cross( glm::vec3(a.x,a.y,a.z), glm::vec3(b.x,b.y,b.z));
	  if (n.z<-0.0001f)
		  t.backfacing = true;
	  else
		  t.backfacing = false;
	  
	  primitives[index] = t;
  }
}

//TODO: Implement a rasterization method, such as scanline.
__global__ void rasterizationKernel(triangle* primitives, int primitivesCount, fragment* depthbuffer, glm::vec2 resolution){
  int index = (blockIdx.x * blockDim.x) + threadIdx.x;
  if(index<primitivesCount){
  triangle t = primitives[index];
  glm::vec3 aabbMin;
  glm::vec3 aabbMax;

  getAABBForTriangle(t,aabbMin,aabbMax);
  int minX = int(floor(aabbMin.x));
  int maxX = int(ceil(aabbMax.x));
  int minY = int(floor(aabbMin.y));
  int maxY = int(ceil(aabbMax.y));

	  for(int j=minY; j<maxY; ++j)
		    for(int i=minX; i<maxX ; ++i)
	  {
		  if ( i>=resolution.x || j >= resolution.y)
			  continue;


		  glm::vec3 bCoords =  calculateBarycentricCoordinate(t,glm::vec2(i,j));

		  if (bCoords.x+bCoords.y+bCoords.z>1.00001f)
			  continue;
		  if (bCoords.x <0.00001f|| bCoords.y<0.00001f || bCoords.z<0.00001f)
			  continue;
		
		  fragment f;
		  f.color = bCoords.x*t.c0 + bCoords.y*t.c1 + bCoords.z*t.c2;
		  f.position = bCoords.x*t.p0world + bCoords.y*t.p1world + bCoords.z*t.p2world;
		  f.normal = bCoords.x*t.n0 + bCoords.y*t.n1 + bCoords.z*t.n2;
		  f.depth = bCoords.x*t.p0.z + bCoords.y*t.p1.z + bCoords.z*t.p2.z;
		  int pixelIndex = i + (j * resolution.x);
		  if ( f.depth < depthbuffer[pixelIndex].depth)
		  {
			  depthbuffer[pixelIndex] = f;
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution, glm::vec3 lightPos,int mode, glm::vec3 camPos){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  fragment f = depthbuffer[index];

	  glm::vec3 col = glm::vec3(0.42f,0.35f,0.80f);
	  if(mode==LIGHTING)
	  {
		glm::vec3 normalizedLP = glm::normalize(lightPos - f.position);
		float lighting = glm::dot(normalizedLP,f.normal);
		depthbuffer[index].color = lighting*col;
	  }
	  if(mode==SLIGHTING)
	  {
		float exponent = 4.0f;
		glm::vec3 normalizedLP = glm::normalize(lightPos - f.position);
		glm::vec3 R = 2*glm::dot(normalizedLP,f.normal)*f.normal - normalizedLP;
		glm::vec3 V = glm::normalize( camPos - f.position);
		float rdotV = max(glm::dot(R,V),0.0f);
		rdotV = min(1.0f, rdotV);
		float difLighting = glm::dot(normalizedLP,f.normal);
		if (f.depth < 10000)
			depthbuffer[index].color = difLighting*col + powf(rdotV,exponent);
	  }
	  else if (mode == COLOR)
	  {
		depthbuffer[index].color = f.color;
	  }
	  else if (mode == NORMALS)
	  {
		depthbuffer[index].color = glm::abs(f.normal);
	  }
	  else if (mode == ZDEPTH)
	  {
	  float d = f.depth>9000?1:f.depth;
	  depthbuffer[index].color = glm::vec3(1-d);
	  }

  }

}

//Writes fragment colors to the framebuffer
__global__ void render(glm::vec2 resolution, fragment* depthbuffer, glm::vec3* framebuffer){

  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  int findex = (resolution.x-1-x) + ( (resolution.y-1-y) * resolution.x);

  if(x<=resolution.x && y<=resolution.y){
    framebuffer[findex] = depthbuffer[index].color;
  }
}

struct is_backfacing
{
	__device__ bool operator() (const triangle t)
	{
		return t.backfacing;
	}
};


// Wrapper for the __global__ call that sets up the kernel calls and does a ton of memory management
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize,const glm::mat4& modelMatrix,const glm::mat4& viewMatrix,const glm::mat4& projectionMatrix,const glm::vec2 zplanes, shadeMode sm, const glm::vec3 camPos){

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
  frag.position = glm::vec3(0,0,0);
  frag.depth = 10000.0;
  clearDepthBuffer<<<fullBlocksPerGrid, threadsPerBlock>>>(resolution, depthbuffer,frag);

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

  device_vbo_worldspace = NULL;
  cudaMalloc((void**)&device_vbo_worldspace, vbosize*sizeof(float));
  cudaMemcpy( device_vbo_worldspace, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_nbo = NULL;
  cudaMalloc((void**)&device_nbo, nbosize*sizeof(float));
  cudaMemcpy( device_nbo, nbo, nbosize*sizeof(float), cudaMemcpyHostToDevice);

  device_cbo = NULL;
  cudaMalloc((void**)&device_cbo, cbosize*sizeof(float));
  cudaMemcpy( device_cbo, cbo, cbosize*sizeof(float), cudaMemcpyHostToDevice);

  tileSize = 32;
  int primitiveBlocks = ceil(((float)vbosize/3)/((float)tileSize));

  //------------------------------
  //vertex shader
  //------------------------------

  glm::vec3 rotAxis (0.0,0.0,1.0);
  cudaMat4 mMatrix = utilityCore::glmMat4ToCudaMat4(modelMatrix);
  cudaMat4 mvpMatrix = utilityCore::glmMat4ToCudaMat4(projectionMatrix*viewMatrix*modelMatrix);
  glm::mat4 h_mInvTranMatrix = glm::transpose(glm::inverse(modelMatrix));
  cudaMat4 mInvTranMatrix = utilityCore::glmMat4ToCudaMat4(h_mInvTranMatrix);
  
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo,device_vbo_worldspace, vbosize,device_nbo, nbosize,mMatrix, mvpMatrix, mInvTranMatrix,resolution,zplanes);
  

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  cudaMat4 vMatrix = utilityCore::glmMat4ToCudaMat4(viewMatrix);
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_worldspace, vbosize, device_cbo, cbosize, device_ibo, ibosize,device_nbo,nbosize, primitives,vMatrix);


  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  thrust::device_ptr<triangle> thrustPrimitivesArray = thrust::device_pointer_cast(primitives);
  unsigned int numberOfPrimitives = ceil((float)ibosize/3);
  int numberOfFrontPrimitives = numberOfPrimitives;
  int rasterizationBlocks = ceil(((float)numberOfFrontPrimitives)/((float)tileSize));
  
  rasterizationKernel<<<rasterizationBlocks, tileSize>>>(primitives,numberOfFrontPrimitives, depthbuffer, resolution);

  cudaDeviceSynchronize();
  //------------------------------
  //fragment shader
  //------------------------------
  glm::vec3 lightPosModel(2,2,2);
  fragmentShadeKernel<<<fullBlocksPerGrid, threadsPerBlock>>>(depthbuffer, resolution,glm::vec3(lightPosModel.x,lightPosModel.y,lightPosModel.z),sm,camPos);

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
  cudaFree( device_vbo_worldspace);
  cudaFree(device_nbo);
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

