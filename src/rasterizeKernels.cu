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
float* device_vbo_viewspace;
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
__global__ void vertexShadeKernel(float* vbo,float* vboViewSpace, int vbosize,  float* nbo, int nbosize,cudaMat4 mvMatrix, cudaMat4 mvpMatrix, cudaMat4 mvpInverseTransMatrix,glm::vec2 resolution,glm::vec2 zplanes){
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
	  viewport.z = glm::vec4(0.0f,0.0f,0.5f*(zplanes.y-zplanes.x),0.5f*(zplanes.y-zplanes.x));
	  viewport.w = glm::vec4(0.0f,0.0f,0.0f,1.0f);
	  
	  transformedVec.w = 1;
	  transformedVec = multiplyMV(viewport,transformedVec);

	  vbo[3*index] = transformedVec.x;
	  vbo[3*index + 1] = transformedVec.y;
	  vbo[3*index + 2] = transformedVec.z;

	  transformedVec = multiplyMV(mvMatrix,v);
	  vboViewSpace[3*index] = transformedVec.x;
	  vboViewSpace[3*index + 1] = transformedVec.y;
	  vboViewSpace[3*index + 2] = transformedVec.z;

	  glm::vec4 n( nbo[3*index], nbo[3*index + 1], nbo[3*index+2],0.0f);
	  glm::vec4 transformedNormal = multiplyMV(mvpInverseTransMatrix,n);
	  nbo[3*index]   = transformedNormal.x;
	  nbo[3*index+1] = transformedNormal.y;
	  nbo[3*index+2] = transformedNormal.z;
  
  }
}

//TODO: Implement primative assembly
__global__ void primitiveAssemblyKernel(float* vbo,float* vboViewSpace, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize,float *nbo, int nbosize, triangle* primitives){
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
	  t.p0view = glm::vec3(vboViewSpace[3*firstVertexIdx], vboViewSpace[3*firstVertexIdx + 1], vboViewSpace[3*firstVertexIdx+2]);
	  t.p1view = glm::vec3(vboViewSpace[3*secondVertexIdx], vboViewSpace[3*secondVertexIdx + 1], vboViewSpace[3*secondVertexIdx+2]);
	  t.p2view = glm::vec3(vboViewSpace[3*thirdVertexIdx], vboViewSpace[3*thirdVertexIdx + 1], vboViewSpace[3*thirdVertexIdx+2]);
	  
	  glm::vec3 n = glm::cross( (t.p1view - t.p0view), (t.p2view - t.p0view));
	  if (n.z<-0.001f)
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

		  ////Referred this for point in triangle test
		  ////http://www.gamedev.net/topic/295943-is-this-a-better-point-in-triangle-test-2d/

		  //triangle t1;
		  //t1.p0.x = i;
		  //t1.p0.y = j;
		  //t1.p1.x = t.p0.x;
		  //t1.p1.y = t.p0.y;
		  //t1.p2.x = t.p1.x;
		  //t1.p2.y = t.p1.y;
		
		  //if (calculateSignedArea(t1)>0.0f)
			 // continue;

		  //t1.p1.x = t.p1.x;
		  //t1.p1.y = t.p1.y;
		  //t1.p2.x = t.p2.x;
		  //t1.p2.y = t.p2.y;
		
		  //if (calculateSignedArea(t1)>0.0f)
			 // continue;
		  //t1.p1.x = t.p2.x;
		  //t1.p1.y = t.p2.y;
		  //t1.p2.x = t.p0.x;
		  //t1.p2.y = t.p0.y;
		
		  //if (calculateSignedArea(t1)>0.0f)
			 // continue;

		  glm::vec3 bCoords =  calculateBarycentricCoordinate(t,glm::vec2(i,j));

		  if (bCoords.x+bCoords.y+bCoords.z>1.00001f)
			  continue;
		  if (bCoords.x <0.00001f|| bCoords.y<0.00001f || bCoords.z<0.00001f)
			  continue;
		  fragment f;
		  f.color = bCoords.x*t.c0 + bCoords.y*t.c1 + bCoords.z*t.c2;
		  //float fragZ = getZAtCoordinate(bCoords,t);
		  f.position = bCoords.x*t.p0view + bCoords.y*t.p1view + bCoords.z*t.p2view;
		  f.normal = bCoords.x*t.n0 + bCoords.y*t.n1 + bCoords.z*t.n2;
		  //f.position.z = fragZ;
		  int pixelIndex = i + (j * resolution.x);
		  if ( f.position.z > depthbuffer[pixelIndex].position.z)
		  {
			  depthbuffer[pixelIndex] = f;
		  }
	  }
  }
}

//TODO: Implement a fragment shader
__global__ void fragmentShadeKernel(fragment* depthbuffer, glm::vec2 resolution){
  int x = (blockIdx.x * blockDim.x) + threadIdx.x;
  int y = (blockIdx.y * blockDim.y) + threadIdx.y;
  int index = x + (y * resolution.x);
  if(x<=resolution.x && y<=resolution.y){
	  fragment f = depthbuffer[index];
	  glm::vec3 lightPos (2,2,2);

	  float lighting = glm::dot(lightPos-f.position,f.normal);
	  depthbuffer[index].color = f.color;
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
void cudaRasterizeCore(uchar4* PBOpos, glm::vec2 resolution, float frame, float* vbo, int vbosize, float* cbo, int cbosize, int* ibo, int ibosize, float* nbo, int nbosize,const glm::mat4& modelMatrix,const glm::mat4& viewMatrix,const glm::mat4& projectionMatrix,const glm::vec2 zplanes){

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

  device_vbo_viewspace = NULL;
  cudaMalloc((void**)&device_vbo_viewspace, vbosize*sizeof(float));
  cudaMemcpy( device_vbo_viewspace, vbo, vbosize*sizeof(float), cudaMemcpyHostToDevice);

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
  cudaMat4 mvMatrix = utilityCore::glmMat4ToCudaMat4(viewMatrix*modelMatrix);
  cudaMat4 mvpMatrix = utilityCore::glmMat4ToCudaMat4(projectionMatrix*viewMatrix*modelMatrix);
  glm::mat4 h_mvInvTranMatrix = glm::transpose(glm::inverse(viewMatrix*modelMatrix));
  cudaMat4 mvInvTranMatrix = utilityCore::glmMat4ToCudaMat4(h_mvInvTranMatrix);
  
  vertexShadeKernel<<<primitiveBlocks, tileSize>>>(device_vbo,device_vbo_viewspace, vbosize,device_nbo, nbosize,mvMatrix, mvpMatrix, mvInvTranMatrix,resolution,zplanes);
  
  ///*TEST VERTEX SHADER*/
  //float* h_vertices = new float[vbosize];
  //cudaMemcpy(h_vertices,device_vbo_viewspace,vbosize*sizeof(float),cudaMemcpyDeviceToHost);
  //for(int i=0; i<vbosize/3;i++)
  //{
	 // std::cout << vbo[3*i] <<","<<vbo[3*i+1]<<","<<vbo[3*i+2]<<std::endl;
	 // //std::cout << [3*i] <<","<<vbo[3*i+1]<<","<<vbo[3*i+2]<<std::endl;
	 // std::cout << h_vertices[3*i] <<","<<h_vertices[3*i+1]<<","<<h_vertices[3*i+2]<<std::endl<<std::endl;
  //}

  //std::cout<<std::endl;
	 //float* h_normals = new float[nbosize];
	 //  cudaMemcpy(h_normals,device_nbo,nbosize*sizeof(float),cudaMemcpyDeviceToHost);

  //for(int i=0; i<nbosize/3;i++)
  //{
	 //std::cout << vbo[3*i] <<","<<vbo[3*i+1]<<","<<vbo[3*i+2]<<std::endl;
	 // std::cout << nbo[3*i] <<","<<nbo[3*i+1]<<","<<nbo[3*i+2]<<std::endl;
	 // std::cout << h_normals[3*i] <<","<<h_normals[3*i+1]<<","<<h_normals[3*i+2]<<std::endl<<std::endl;
  //}
  //std::cout<<std::endl<<std::endl;

  cudaDeviceSynchronize();
  //------------------------------
  //primitive assembly
  //------------------------------
  primitiveBlocks = ceil(((float)ibosize/3)/((float)tileSize));
  primitiveAssemblyKernel<<<primitiveBlocks, tileSize>>>(device_vbo, device_vbo_viewspace, vbosize, device_cbo, cbosize, device_ibo, ibosize,device_nbo,nbosize, primitives);

  /*TEST PRIMITIVE ASSEMBLY*/
  //triangle* h_triangles = new triangle[ibosize/3];
  //cudaMemcpy(h_triangles,primitives,ibosize/3*sizeof(triangle),cudaMemcpyDeviceToHost);
  //
  //for(int i=0; i <ibosize/3 ; i++)
  //{
	 // triangle t = h_triangles[i];
	 // glm::vec3 n = glm::normalize(glm::cross( (t.p1view - t.p0view), (t.p2view-t.p0view)));
	 // std::cout<<n.x<<","<<n.y<<","<<n.z<<"  "<<t.backfacing<<std::endl;
  //}

  cudaDeviceSynchronize();
  //------------------------------
  //rasterization
  //------------------------------
  thrust::device_ptr<triangle> thrustPrimitivesArray = thrust::device_pointer_cast(primitives);
  unsigned int numberOfPrimitives = ceil(ibosize/3);
  int numberOfFrontPrimitives = thrust::remove_if(thrustPrimitivesArray,thrustPrimitivesArray+numberOfPrimitives,is_backfacing()) - thrustPrimitivesArray ;
  int rasterizationBlocks = ceil(((float)numberOfFrontPrimitives)/((float)tileSize));
  
  rasterizationKernel<<<rasterizationBlocks, tileSize>>>(primitives,numberOfFrontPrimitives, depthbuffer, resolution);

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
  cudaFree( device_vbo_viewspace);
  cudaFree( device_cbo );
  cudaFree( device_ibo );
  cudaFree( framebuffer );
  cudaFree( depthbuffer );
}

